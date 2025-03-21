import copy
import gc
import itertools
import json
import os
import warnings
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from math import floor, ceil
from omegaconf import OmegaConf
from transformers import BertConfig
from transformers.activations import ACT2FN

import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.utils.visualizer import Visualizer
from lavis.common.registry import registry
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY as detrectron_registry
from lavis.models.sggp_models import SggpBase
from lavis.models.sggp_models.sggp_outputs import (
    SggpIntermediateOutput,
    SggpOutputWithLogits,
)
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.med import XBertEncoder, BertLMHeadModel, BertForMaskedLM, BertOnlyMLMHead, UnifiedBertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from detectron2.modeling.backbone import SimpleFeaturePyramid, ViT
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import Instances, Boxes, ImageList
from detectron2.utils.events import EventStorage
from lavis.common.utils import get_abs_path
from torch.nn import CrossEntropyLoss
# from lavis.datasets.datasets.vg_sgg_datasets import BoxList


from lavis.common.masking_generator import InteractMaskingGenerator
from detectron2.layers.roi_align import ROIAlign
import math
from lavis.processors.sggp_processors import RelationSampling
from detectron2.structures import BoxMode
from detectron2.modeling import StandardROIHeads
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.layers import Conv2d, get_norm, ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
#目标检测gt：[self.idx_to_label[str(int(i))] for i in samples["labels"][0].get_field("labels")]
#预测：detect_results[0]['phrases'] （使用predict作为api）
from groundingdino.util.inference import load_model, load_image, predict, annotate

from pysgg.structures.image_list import to_image_list

from ..object_detector.backbone import build_backbone
from ..object_detector.roi_heads.relation_head.utils_motifs import to_onehot
from ..object_detector.rpn.rpn import build_rpn
from ..object_detector.roi_heads.roi_heads import build_roi_heads
from pysgg.config.defaults import  _C as detection_cfg
from ..object_detector.roi_heads.relation_head.inference import make_roi_relation_post_processor
def find_matches(dict1, dict2):
    matches = {}
    for key, value in dict1.items():
        if key.split()[0] in dict2:
            matches[value] = dict2[key.split()[0]]
    return matches

def vocab_match_predicate(dict1, dict2):
    '''
    dict1: vocab
    dict2: 51 predicate
    key是vocab,
    '''
    matched_dict = {}
    for key2,value2 in dict2.items():
        first_word = key2.split()[0]
        for key1 in dict1.keys():
            if first_word in key1:
                matched_dict[value2] = dict1[key1]
                break
    return matched_dict
@registry.register_model("SggpFinetune")
class SggpFinetune(SggpBase):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_cc3m": "configs/models/sggp_cc3m.yaml",
    }


    def __init__(
        self,
        detector,
        unified_encoder,
        num_classes,
        fpn_config,
        proposal_generator,
        roi_heads,
        momentum=0.995,
        alpha=0.4,
        RelationSampler = None,
        relation_post_processor = None,
        max_txt_len=30,
        mode="bert",# or "zero-shot" "contrastive"
        train_detector_only= False,
        task_mode="sgdet",
        window_size=0
    ):
        super().__init__()
        # BERTCONFIG = "configs/models/sggp_text_bert.json"  # todo 从配置文件中读入
        # BERTCONFIG = BertConfig.from_json_file(get_abs_path(BERTCONFIG))
        self.tokenizer = self.init_tokenizer()
        self.train_detector_only = train_detector_only
        self.max_txt_len = max_txt_len
        self.detector = detector
        self.predict_mode = mode
        self.unified_encoder = unified_encoder
        cfg_file=OmegaConf.load("lavis/configs/datasets/vg/defaults_sgg.yaml")
        vocab_file=json.load(open(os.path.join("cache",cfg_file.datasets.visual_genome.build_info.annotations.train.storage[1])))
        self.idx_to_label =vocab_file['idx_to_label']
        self.predicate_to_idx =vocab_file['predicate_to_idx']
        self.idx_to_predicate = vocab_file['idx_to_predicate']
        self.predicate_to_idx['background'] = 0
        self.idx_to_predicate['0'] = 'background'
        self.map_predicate2bert_idx = find_matches(self.predicate_to_idx, self.tokenizer.vocab)
        torch.save(self.map_predicate2bert_idx, 'map_predicate2bert_idx.pth')
        self.bert_idx2map_predicate = vocab_match_predicate(self.tokenizer.vocab,self.predicate_to_idx)
        hidden_size = unified_encoder.config.hidden_size
        self.patch_size = unified_encoder.num_patches
        # self.text_encoder.encoder.gradient_checkpointing=True
        # self.decoder.bert.encoder.gradient_checkpointing = True
        self.caption = "airplane . animal . arm . bag . banana . basket . beach . bear . bed . bench . bike . bird . board . boat . book . boot . bottle . bowl . box . boy . branch . building . bus . cabinet . cap . car . cat . chair . child . clock . coat . counter . cow . cup . curtain . desk . dog . door . drawer . ear . elephant . engine . eye . face . fence . finger . flag . flower . food . fork . fruit . giraffe . girl . glass . glove . guy . hair . hand . handle . hat . head . helmet . hill . horse . house . jacket . jean . kid . kite . lady . lamp . laptop . leaf . leg . letter . light . logo . man . men . motorcycle . mountain . mouth . neck . nose . number . orange . pant . paper . paw . people . person . phone . pillow . pizza . plane . plant . plate . player . pole . post . pot . racket . railing . rock . roof . room . screen . seat . sheep . shelf . shirt . shoe . short . sidewalk . sign . sink . skateboard . ski . skier . sneaker . snow . sock . stand . street . surfboard . table . tail . tie . tile . tire . toilet . towel . tower . track . train . tree . truck . trunk . umbrella . vase . vegetable . vehicle . wave . wheel . window . windshield . wing . wire . woman . zebra ."  #"airplane animal arm bag banana basket beach bear bed bench bike bird board boat book boot bottle bowl box boy branch building bus cabinet cap car cat chair child clock coat counter cow cup curtain desk dog door drawer ear elephant engine eye face fence finger flag flower food fork fruit giraffe girl glass glove guy hair hand handle hat head helmet hill horse house jacket jean kid kite lady lamp laptop leaf leg letter light logo man men motorcycle mountain mouth neck nose number orange pant paper paw people person phone pillow pizza plane plant plate player pole post pot racket railing rock roof room screen seat sheep shelf shirt shoe short sidewalk sign sink skateboard ski skier sneaker snow sock stand street surfboard table tail tie tile tire toilet towel tower track train tree truck trunk umbrella vase vegetable vehicle wave wheel window windshield wing wire woman zebra ."

        self.BOX_TRESHOLD = 0.25
        self.TEXT_TRESHOLD = 0.2
        self.window_size=window_size

        #fpn module
        self.stages = fpn_config["stages"]
        add_modules = fpn_config["add_module"]
        self._out_feature_strides = fpn_config["out_feature_strides"]
        self.strides =  fpn_config["strides"]
        self.out_channels =  fpn_config["out_channels"]
        self.top_block = LastLevelMaxPool()
        #add_modules is similar to fpn, e.g., plainfeatureprimid
        for stage,layer in  add_modules.items():
            self.add_module(stage,layer)
        if self.top_block is not None:#非none，则在stages的基础上再加一个top_block
            last_stage = int(list(self._out_feature_strides)[-1][-1]) #5
            for s in range(last_stage,last_stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: self.out_channels for k in
                                      self._out_features}  # {'p2': 256, 'p3': 256, 'p4': 256, 'p5': 256, 'p6': 256}
        _size_divisibility = self.strides[-1]
        _square_pad = 0 #todo


        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.relation_post_processor = relation_post_processor

        if mode =="bert":
            #for linear probing
            # self.rel_cls_head = nn.Linear(hidden_size, 51)

            self.loss_bert = CrossEntropyLoss()  # -100 index = not masked tocken
        if mode == "contrastive":

            self.interactive_masking = InteractMaskingGenerator(1024,16,120000)
            self.vision_proj = nn.Linear(768, 768)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_bert = CrossEntropyLoss()

        if num_classes > 0:
            head_config = copy.deepcopy(unified_encoder.config)
            head_config.hidden_size = hidden_size * 2
            # self.cls_head =  BertOnlyMLMHead(head_config)
        else:
            warnings.warn(
                f"Found num_classes=0, initializing {type(self)} without classifier."
            )
        # mode
        self.task_mode = task_mode
        self.relation_sampler = RelationSampler





    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True,demo=False):
        if demo==False:
            # for relation gt and image preparation
            instance_gts = []
            for label in samples["low_res_target"]:
                instance_gts.append(conver_boxlist_to_instance(label).to("cuda"))

            det_gts = []
            for label in samples["labels"]:
                det_gts.append(conver_boxlist_to_instance(label).to("cuda"))


        image = samples["image"]

        images = ImageList.from_tensors(
            image,
            0,
            # padding_constraints={'size_divisiblity': 32, 'square_size': 1024},
        )

        if self.task_mode == "sgdet":
            with torch.no_grad():
                self.detector.eval()
                samples["labels"] = [label.to("cuda") for label in  samples["labels"]]
                features = self.detector.backbone(images.tensor)
                proposals, _ = self.detector.rpn(images, features, samples["labels"])  # targets:box_list.proposals:仅有objectness
                x, predictions, _ = self.detector.roi_heads(features,  proposals, samples["labels"])
                predictions =   [conver_boxlist_to_instance(prediction) for prediction in predictions]
                # self.metadata = MetadataCatalog.get(
                #         "visual_genome")
                # vis = Visualizer(images.tensor[0].permute(1,2,0).cpu()*255) #metadata
                # vis_pred = vis.draw_instance_predictions(predictions[0].to("cpu")).get_image()
                # plt.imshow(vis_pred)
                # plt.show()
                if is_train:
                    (  # prepare sub-obj pair
                     predictions,
                     rel_labels,
                     rel_labels_all,
                     rel_pair_idxs,
                     gt_rel_binarys_matrix,
                     ) = self.relation_sampler.detect_relsample(predictions, det_gts)
                    relation_targets = rel_labels_all
                else:
                    rel_pair_idxs = self.relation_sampler.prepare_test_pairs(  # 不超过设定的max relation数，就全部保留
                        torch.device("cuda"), predictions
                    )
                    relation_targets= [None for i in range(len(rel_pair_idxs))]


        else:
            predictions = [] # for predcls
        low_res_imgs = samples["low_res_img"]

        low_res_imgs = ImageList.from_tensors(
            low_res_imgs,
            0,
            padding_constraints={'size_divisiblity': 32, 'square_size': 224},
        )

        # plt.imshow((images.tensor[0].cpu().permute(1,2,0).numpy()))
        # plt.show()




        # instance_gts = [instance_gt.to(self.device) for instance_gt in instance_gts]
        detrectron_data=[]


        # strack image list to bs tensor
        # image=torch.stack(samples["image"], 0)
        # image_embeds = self.detector.backbone.net(image)['last_feat']
        # if self.task_mode == "predcls":
        #     relation_targets = []
        #     for i in range(len(samples["index"])):
        #
        #         relation_targets.append(instance_gts[i].get("relation_tuple")[:, -1].to(self.device))
        #         # for vitdet
        #         meta = {}
        #         for key, value in samples.items():
        #             if isinstance(value,int):
        #                 continue
        #             meta[key]=value[i]
        #             meta["instances"]=det_gts[i]
        #         # relation_targets=torch.cat(relation_targets,0).to(self.device)
        #         detrectron_data.append(meta)

        loss_dict = {}
        # get detector's result
        image_det = []
        for i in image:
            image_det.append({"image":i})


        if demo!=True:

            samples["sentance_tockens"] = []
            samples["relation_index"] = []
            samples["negative_text_sets"] = []
            interactive_boxes = []
            if self.task_mode == "sgdet":  # predict object
                # images=torch.stack(image, 0)
                text_inputs = ['object detection [SEP]' for i in range(len(images))]

                tokenized_texts = self.tokenizer(
                    text_inputs,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)
                # with EventStorage() as storage:
                #     loss_dict,vit_output=self.detector(detrectron_data)
                # for detection training, use same encoder as my model.
                # with EventStorage() as storage:
                #
                #      outputs = self.unified_encoder(  # BertModel
                #         input_images=images.tensor,
                #         input_text_ids=None, #目标检测不需要text.tokenized_texts.input_ids,
                #         text_attention_mask=None,#tokenized_texts.attention_mask,
                #         mode=self.predict_mode,
                #         gt_instances=det_gts,
                #         window_size=self.window_size
                #      )
                #      # stage实现vitdet中的plainpirid
                #      detection_results = []
                #      img_hidden_states = outputs.last_hidden_state
                #      img_hidden_states = rearrange(img_hidden_states, 'b (h w) c -> b c h w ',
                #                                    h=int(math.sqrt(img_hidden_states.shape[1])))
                #      for stage in self.stages:
                #          detection_results.append(stage(img_hidden_states))  # (list:4) (bs,256,h,w)
                #
                #      if self.top_block is not None:
                #          top_block_in_feature = detection_results[-1]  # todo 这里简化了，直接选取最top的特征
                #          detection_results.extend(self.top_block(top_block_in_feature))
                #      assert len(self._out_features) == len(detection_results)
                #      featurePyramid = {f: res for f, res in zip(self._out_features, detection_results)}
                #      # RPN
                #
                #      proposals, proposal_losses = self.proposal_generator(
                #         images, featurePyramid, # ImageList(images, [input_image.shape[-2:] for input_image in images])
                #          det_gts)  # todo 对于bs！= 1，且要padding img的情况 可能需要改变[input_images.shape[-2:]]
                #
                #
                #      if is_train:
                #          loss_dict.update(proposal_losses)# record rpn loss
                #          predictions, detector_losses = self.roi_heads(images,featurePyramid, proposals, det_gts) #ImageList(images, [input_image.shape[-2:] for input_image in images])
                #          loss_dict.update(detector_losses)
                #          # 训练期间仍需要inference,从而获得预测的Box执行sgg部分
                #          predictions_for_inference = [prediction.predictions_for_inference for prediction in predictions]
                #          predictions = predictions_for_inference
                #
                #          #
                #          # (  # prepare sub-obj pair
                #          #     predictions,
                #          #     rel_labels,
                #          #     rel_labels_all,
                #          #     rel_pair_idxs,
                #          #     gt_rel_binarys_matrix,
                #          # ) = self.relation_sampler.detect_relsample(predictions_for_inference, det_gts) #todo 注意此时predction是1024尺度下的
                #          # relation_targets = rel_labels_all  # sgdet的时候，重采样的target代替真实的标注
                #         #由于目标检测从头开始，没有有价值的预测，所以在sgdet任务中，仍然采用类似predcls中的直接采样gt
                #
                #          rel_pair_idxs = []
                #          relation_targets = []
                #
                #          for instance_gt,prediction in zip(instance_gts,predictions):
                #
                #              # prediction.pred_classes=instance_gt.gt_classes
                #              # prediction.pred_boxes = instance_gt.gt_boxes
                #              # prediction.scores = torch.ones((len(instance_gts[0])),device=instance_gt.gt_boxes.device)#tensor(num_prop,):int
                #              # prediction.locating_match= torch.ones((len(instance_gts[0])),device=instance_gt.gt_boxes.device) #tensor(num_prop,):int
                #              rel_pair_idxs.append(instance_gt.get("relation_tuple")[:,:2])
                #              relation_targets.append(instance_gt.get("relation_tuple")[:,-1])
                #
                #      else:
                #          #尽管sgdet训练的时候用的是gt box训练，但inference的时候得用pret box
                #          predictions,_ = self.roi_heads(images,#_ 本身返回的就是空{} #ImageList(images, [images.shape[-2:]])
                #                                                        featurePyramid, proposals, det_gts)
                #
                #
                #          rel_pair_idxs = self.relation_sampler.prepare_test_pairs(#不超过设定的max relation数，就全部保留
                #                 self.device, predictions)
                # use out-of-shelf detector from pysgg


            relation_losses = []
            contrastive_losses = []
            #prepare text input
            if self.task_mode == "predcls":
                if not is_train:
                    rel_pair_idxs = self.relation_sampler.prepare_test_pairs(  # 不超过设定的max relation数，就全部保留
                        torch.device("cuda"), proposals=det_gts
                    )
                    relation_targets = [None for i in range(len(rel_pair_idxs))]
                for idx,instance_gt in enumerate(instance_gts):
                    sentance_tockens = []
                    negative_text_sets = []
                    interactive_box = []
                    label=instance_gt.get("labels")
                    #prepare sgg text prompt
                    if is_train:
                        (  # return的都是list:num_image
                            predictions,  # [num_prop]
                            rel_labels,  # [num_prop*(num_prop-1)]
                            rel_pair_idxs,  # [num_prop*(num_prop-1),2]
                            gt_rel_binarys_matrix,  # [num_prop,num_prop]
                        ) = self.relation_sampler.gtbox_relsample(instance_gt,instance_gt)
                        relation_targets = rel_labels
                        for triplet in rel_pair_idxs[idx]: #instance_gt.get("relation_tuple"):

                            if self.predict_mode == "contrastive":
                                # 准备text prompt
                                if is_train:
                                    get_feature_text_inputs = "get visual interactive feature [SEP] "
                                    text_inputs = " relation between {} and {} is {} ".format(
                                                   self.idx_to_label[str(int(label[int(triplet[0])]))],
                                                       self.idx_to_label[str(int(label[int(triplet[1])]))],self.idx_to_predicate[str(int(triplet[2]))])
                                    # text_inputs = "the relation between two objects are {} ".format(self.idx_to_predicate[str(int(triplet[2]))])

                                    negative_text_set = [f"  relation between {self.idx_to_label[str(int(label[int(triplet[0])]))]} and {self.idx_to_label[str(int(label[int(triplet[1])]))]} is {c} " for c in list(self.predicate_to_idx.keys()) if c != self.idx_to_predicate[str(int(triplet[2]))] ]
                                    # negative_text_set = random.sample(negative_text_set, 5)
                                    sentance_tockens.append(text_inputs)
                                    negative_text_sets.append(negative_text_set)
                                else:
                                    get_feature_text_inputs = "get visual interactive feature [SEP] "
                                    text_inputs = [f" relation between {self.idx_to_label[str(int(label[int(triplet[0])]))]} and " \
                                                   f"{ self.idx_to_label[str(int(label[int(triplet[1])]))]} is {r} " for r in
                                                  self.predicate_to_idx.keys()]
                                    # text_inputs = [f"the relation between two objects are {c} " for c in self.predicate_to_idx.keys()]
                                    # text_inputs = " relation between {} and {} is [MASK] ".format(
                                    #     self.idx_to_label[str(int(label[int(triplet[0])]))],
                                    #     self.idx_to_label[str(int(label[int(triplet[1])]))])
                                    sentance_tockens.append(text_inputs)
                                    # sentance_tockens = text_inputs


                            else:

                                text_inputs = 'relation prediction [SEP]  ' +  self.idx_to_label[
                                    str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                                                  str(int(label[int(triplet[1])]))]
                                sentance_tockens.append(text_inputs)


                                    # temp = 'the scene of  :' + self.idx_to_label[
                                #     str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                                #            str(int(label[int(triplet[1])]))]
                                # temp =  self.idx_to_label[
                                #     str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                                #            str(int(label[int(triplet[1])]))]

                            if self.predict_mode == "contrastive":
                                ratio_scale = (224 / instance_gt.image_size[0], 224 / instance_gt.image_size[1])

                                box1 = instance_gt.gt_boxes[int(triplet[0])].tensor[0]#0为了从(1,4)->(4,)
                                box2 = instance_gt.gt_boxes[int(triplet[1])].tensor[0]
                                top = floor((max(box1[1], box2[1])) / 16 * ratio_scale[1])
                                left = floor((max(box1[0], box2[0])) / 16*ratio_scale[0])
                                botten = ceil((min(box1[3], box2[3])) / 16 * ratio_scale[1])
                                right = ceil((min(box1[2], box2[2])) /  16*ratio_scale[0])


                                if left>right:
                                    temp = left
                                    left = right
                                    right = temp
                                if top>botten:
                                    temp = top
                                    top = botten
                                    botten = temp
                                if (right - left) == 0:
                                    right = right + 1
                                if (botten - top) == 0:
                                    botten = top + 1

                                union_box = torch.tensor((left, top, right, botten),dtype=torch.long)
                                interactive_box.append(union_box)
                    else:

                        for triplet in rel_pair_idxs[idx]:
                            text_inputs = 'relation prediction [SEP]  ' + self.idx_to_label[
                                str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' +' ' + self.idx_to_label[
                                              str(int(label[int(triplet[1])]))]
                            sentance_tockens.append(text_inputs)

                    if self.predict_mode == "contrastive":
                        interactive_boxes.append(torch.stack(interactive_box))
                        samples["negative_text_sets"].append(negative_text_sets)
                        # for c in self.predicate_to_idx.keys():
                        #     sentance_tockens.append("the relation between two objects are {} ".format(c) )
                    # interactive_boxes.append(interactive_box)

                    samples["sentance_tockens"].append(sentance_tockens)
                    # samples["relation_index"].append(relation_index)
            else:
                #当训练的时候,predictions,rel_pair_idxs已经变成ginstance_gts的了
                for rel_pair_idx, prediction in zip(rel_pair_idxs, predictions):
                    sentance_tockens = []
                    obj_class = prediction.pred_classes
                    for pair in rel_pair_idx:
                        try:
                            text_inputs = 'relation prediction [SEP]  ' + 'the ' + self.idx_to_label[
                            str(int(obj_class[int(pair[0])]))] + ' '  + '[MASK]'  +' ' + '[MASK]'  +' ' + 'the '+ self.idx_to_label[
                                          str(int(obj_class[int(pair[1])]))]
                        except:
                            a = 1
                        sentance_tockens.append(text_inputs)
                    samples["sentance_tockens"].append(sentance_tockens)

            if not self.train_detector_only:
                pred = [] #used for predcls, zero-shot
                for i,(image,low_res_img,sentance_tocken,batch_relation_targets,instance_gt) in enumerate(zip(images,low_res_imgs.tensor, samples["sentance_tockens"],relation_targets,instance_gts)):
                    logits_per_image = []
                    labels_per_image = []
                    # prediction = []

                    if  (self.predict_mode != "contrastive" or is_train) == True:
                        try:
                            tokenized_texts = self.tokenizer(
                            sentance_tocken,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(self.device)
                        except:
                            a=1

                        try:
                            # relation_index = torch.where(tokenized_texts["input_ids"] == 103)[1]  # relation_index是relation词出现的位置 #todo 考虑batch中可能relation出现位置不一致,这先按照一致计算
                            relation_index = torch.where(tokenized_texts["input_ids"] == 103)[1].view(#如果有多个mask，选第一个
                                tokenized_texts["input_ids"].shape[0], -1)[:, 0]
                            if len(relation_index)==0:
                                relation_index = [8]
                        except:
                            relation_index = [8]

                    if  self.predict_mode != "contrastive":

                        encoder_output = self.unified_encoder(  # BertModel
                        input_images=low_res_img.repeat(len(tokenized_texts['input_ids']),1,1,1),#todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                        input_text_ids=tokenized_texts.input_ids,
                        text_attention_mask=tokenized_texts.attention_mask,
                        mode=self.predict_mode)

                    if self.predict_mode == "zero-shot":
                        if is_train:
                            prediction.append(encoder_output.logits[:, relation_index, :])
                            relation_loss = self.loss_bert(
                                encoder_output.logits[:,relation_index,:].view(-1, self.unified_encoder.config.vocab_size),
                                torch.tensor([self.map_predicate2bert_idx[int(item)] for item in batch_relation_targets.view(-1)],dtype=torch.long).cuda())

                            relation_losses.append(relation_loss)

                        else:
                            if self.task_mode == "predcls":
                                pred.append(encoder_output.logits[torch.arange(encoder_output.logits.shape[0]), relation_index, :])
                            else:
                                predictions[i].set('relation_logits',encoder_output.logits[torch.arange(encoder_output.logits.shape[0]), relation_index, :])

                    if self.predict_mode == "bert":
                        pred = []
                        # 用于更换unify encoder里的bert_cls情况下的训练，sggp_unified_bert.json 的decoder_vocab_size设置为51
                        for b, index in enumerate(relation_index):
                            pred.append(encoder_output.logits[b, index, :].unsqueeze(0))
                        pred = torch.cat(pred,0)
                        # for linear probing
                        # pred = self.rel_cls_head(encoder_output.hidden_states[-1][:, 197+relation_index, :])
                        if is_train:
                            relation_loss = self.loss_bert(
                                pred.view(-1, self.unified_encoder.config.decoder_vocab_size),
                                batch_relation_targets.view(-1))


                            relation_losses.append(relation_loss)

                        if is_train is False:
                            pred = pred.squeeze(1)
                            # pred = self.postprocessor(pred,instance_gt)#todo 只适用于51类预测!,且只能用于predcls
                            if self.task_mode == "predcls":
                                predictions.append(pred)
                            else:
                                predictions[i].set("rel_scores",pred)
                                #record proposal prediction
                                predictions[i].set("rel_pair_idxs", rel_pair_idxs[i])
                                # predictions.append( predictions[i])

            # del samples, image_embeds, instance_gts
            if is_train:


                if self.train_detector_only==False:
                    relation_losses = sum(relation_losses) / len(samples["image"])
                    loss_dict["relation_loss"] = relation_losses
                loss_dict["loss"] = sum(loss_dict.values())
                return loss_dict

            else:
                if self.task_mode == "predcls":
                    obj_labels = [detgt.get("labels") for detgt in det_gts]
                    predict_logits = [to_onehot(obj_label,151) for obj_label in obj_labels] #todo 注意这里显式设置了目标检测数目
                    # pred_scores =  torch.ones(len(obj_labels)).to("cuda")
                    # pred_labels = obj_labels.to("cuda")
                    if self.predict_mode == "zero-shot":
                        predictions = self.relation_post_processor(
                            (pred, predict_logits, None),
                            rel_pair_idxs, det_gts,zeroshot=True)
                    else:
                        predictions = self.relation_post_processor(
                            (pred, predict_logits, None),
                            rel_pair_idxs, det_gts)
                    pred = [prediction.get("rel_scores")for prediction in predictions]
                    return {"predictions": pred, "instance_gts":predictions}
                else:
                    if self.predict_mode == "zero-shot":

                        # if self.task_mode == "predcls":
                        #     predictions.append(
                        #         encoder_output.logits[torch.arange(encoder_output.logits.shape[0]), relation_index,
                        #         :].cpu())
                        # else:
                        predict_logits = [prediction.predict_logits for prediction in predictions]
                        relation_logits =[prediction.relation_logits for prediction in predictions]  # prediction[1] is equal to: encoder_output.logits[torch.arange(encoder_output.logits.shape[0]),relation_index, :].cpu()
                        assert len(relation_logits)==len(predict_logits)==1

                        predictions = self.relation_post_processor(
                            (relation_logits, predict_logits, None),
                            rel_pair_idxs, predictions,zeroshot=True)

                        # predictions[i].set('rel_scores',
                        #                    encoder_output.logits[torch.arange(encoder_output.logits.shape[0]),
                        #                    relation_index, :].squeeze(-2).cpu())
                        # predictions[i].set("rel_pair_idxs", rel_pair_idx)
                    else:
                        predict_logits=[prediction.predict_logits for prediction in predictions]
                        predictions = self.relation_post_processor(
                        (pred, predict_logits, None),
                        rel_pair_idxs, predictions)
                    return {"predictions": predictions, "instance_gts": det_gts} #det_gts是大尺寸gt,和predctions对应,sgdet任务需要对iou评估，所以采用det_gt,且det_gtHe ins_gt的relation部分一致
        else:#demo
            sentance = samples['text_input']
            tokenized_texts = self.tokenizer(
                sentance,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            encoder_output = self.unified_encoder(  # BertModel
                input_images=low_res_imgs[0].repeat(len(tokenized_texts['input_ids']), 1, 1, 1),
                # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                input_text_ids=tokenized_texts.input_ids,
                text_attention_mask=tokenized_texts.attention_mask,
                mode=self.predict_mode)
            mask_idx = (torch.where(tokenized_texts['input_ids']==103)[1])
            return (encoder_output.logits[torch.arange(encoder_output.logits.shape[0]), mask_idx, :])


    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        # -------for initialing detectron detector ---------
        # detector=model_zoo.get_config(cfg.detector_config_path).model
        # image_encoder = VisionTransformerEncoder.from_config(cfg)
        # detector=instantiate(detector)

        # -------for grounding modal -----------
        # CONFIG_PATH = get_abs_path(cfg.detector_config_path)
        # grounding_model = load_model(CONFIG_PATH)
        # -------for faster rcnn modal -----------

        detection_cfg.merge_from_file(cfg.detector_config_path)
        backbone = build_backbone(detection_cfg)
        rpn = build_rpn(detection_cfg, backbone.out_channels)  # 256
        roi_heads = build_roi_heads(detection_cfg, backbone.out_channels)
        detector = nn.ModuleDict({"backbone":backbone,"rpn":rpn,"roi_heads":roi_heads})
        from pysgg.utils.checkpoint import DetectronCheckpointer
        checkpointer = DetectronCheckpointer(
            detection_cfg, detector)


        # text encoder + multimodal encoder
        # text_encoder = XBertEncoder.from_config(cfg)
        # decoder_cfg=BertConfig.from_json_file(get_abs_path(cfg.med_config_path))
        # decoder_cfg.vocab_size=51
        # decoder = BertForMaskedLM(decoder_cfg)
        unified_encoder = UnifiedBertForMaskedLM.from_config(cfg)
        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 30)
        mode = cfg.get("predict_mode", "bert")
        train_detector_only = cfg.get("train_detector_only", False)
        task_mode = cfg.get("task_mode")
        window_size = cfg.get("window_size")
        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )
        RelationSampler = RelationSampling(
            cfg.RELATION.FG_IOU_THRESHOLD,
            cfg.RELATION.REQUIRE_BOX_OVERLAP,
            cfg.RELATION.NUM_SAMPLE_PER_GT_REL,
            cfg.RELATION.BATCH_SIZE_PER_IMAGE,
            cfg.RELATION.POSITIVE_FRACTION,
            cfg.RELATION.MAX_PROPOSAL_PAIR,
            cfg.RELATION.USE_GT_BOX,
            cfg.RELATION.TEST.REQUIRE_OVERLAP,
        )
        #fpn, rpn, roi init
        #fpn
        dim = cfg.encoder_width
        out_channels = cfg.out_channels
        max_txt_len = cfg.max_txt_len
        strides = [int(16 / scale) for scale in cfg.scale_factors] #todo 这里直接采用默认的初始stride，16

        proposal_generator = RPN(OmegaConf.create({"MODEL": cfg}))
        stages = []
        add_module ={}
        use_bias = "LN" == ""
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        _out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in
                                strides}  # {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}

        for idx, scale in enumerate(cfg.scale_factors):

            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm("LN", dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm("LN", out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm("LN", out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))

            add_module["simfp_{}".format(stage)] = layers
            stages.append(layers)
            fpn_config = {"stages": stages, "add_module": add_module,"out_feature_strides":_out_feature_strides,"strides":strides,"out_channels":out_channels}

        roi_heads = StandardROIHeads(cfg)

        relation_post_processor = make_roi_relation_post_processor(cfg)
        model = cls(
            detector=detector,#None, #detector, #grounding_model,
            unified_encoder=unified_encoder,
            num_classes = num_classes,
            fpn_config=fpn_config,
            proposal_generator = proposal_generator,
            roi_heads = roi_heads,
            RelationSampler = RelationSampler,
            relation_post_processor = relation_post_processor,
            alpha=alpha,
            momentum=momentum,
            max_txt_len=max_txt_len,
            mode=mode,
            train_detector_only=train_detector_only,
            task_mode = task_mode,
            window_size = window_size
        )

        model.load_checkpoint_from_config(cfg)
        checkpoint = checkpointer.load(
            detection_cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, update_schedule=False
        )

        return model
'''
creator: cxg
for converting pysgg structure "boxlist" to detrectron's "instances"
'''
def conver_boxlist_to_instance(boxlist):
    structure=Instances(boxlist.size[::-1]) #因为boxlist是(w,h), Instance是(h,w)
    assert boxlist.mode == "xyxy", "mode is not xyxy, so the folllowing convert is wrong"
    try:
        structure.set("gt_classes", boxlist.get_field("labels"))
        gt_boxes=Boxes(boxlist.bbox)
        structure.set("gt_boxes",gt_boxes)
    except:
        pred_boxes = Boxes(boxlist.bbox)
        structure.set("pred_boxes", pred_boxes)
        structure.set("pred_classes", boxlist.get_field("pred_labels"))
        structure.set("scores", boxlist.get_field("pred_scores"))
    for field in boxlist.fields():
        structure.set(field,boxlist.get_field(field))

    return structure


