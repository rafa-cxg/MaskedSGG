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

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
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
# from lavis.processors.sggp_processors import PostProcessor

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
@registry.register_model("SggpFinetunevitdet")
class SggpFinetunevitdet(SggpBase):

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
        max_txt_len=30,
        mode="bert",# or "zero-shot" "contrastive"
        task_mode="sgdet",
        window_size=0
    ):
        super().__init__()
        # BERTCONFIG = "configs/models/sggp_text_bert.json"  # todo 从配置文件中读入
        # BERTCONFIG = BertConfig.from_json_file(get_abs_path(BERTCONFIG))
        self.tokenizer = self.init_tokenizer()

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

        if mode =="bert":
            #for linear probing
            # self.rel_cls_head = nn.Linear(hidden_size, 51)
            self.postprocessor = PostProcessor(True)
            self.loss_bert = CrossEntropyLoss()  # -100 index = not masked tocken
        if mode == "contrastive":
            self.postprocessor = PostProcessor(True)
            self.interactive_masking = InteractMaskingGenerator(1024,16,120000)
            self.vision_proj = nn.Linear(768, 768)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_bert = CrossEntropyLoss()
        if num_classes > 0:
            head_config = copy.deepcopy(unified_encoder.config)
            head_config.hidden_size = hidden_size * 2
            self.cls_head =  BertOnlyMLMHead(head_config)
        else:
            warnings.warn(
                f"Found num_classes=0, initializing {type(self)} without classifier."
            )
        # mode
        self.task_mode = task_mode
        self.relation_sampler = RelationSampler



    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        # for relation gt and image preparation
        instance_gts = []
        for label in samples["low_res_target"]:
            instance_gts.append(conver_boxlist_to_instance(label).to("cuda"))



        det_gts = []
        image = samples["image"]
        for label in samples["labels"]:
            det_gts.append(conver_boxlist_to_instance(label).to("cuda"))


        # instance_gts = [instance_gt.to(self.device) for instance_gt in instance_gts]
        detrectron_data=[]
        relation_targets=[]

        # strack image list to bs tensor
        # image=torch.stack(samples["image"], 0)
        # image_embeds = self.detector.backbone.net(image)['last_feat']
        for i in range(len(samples["index"])):

            relation_targets.append(instance_gts[i].get("relation_tuple")[:, -1].to(self.device))
            # for vitdet
            meta = {}
            for key, value in samples.items():
                if isinstance(value,int):
                    continue
                meta[key]=value[i]
                meta["instances"]=det_gts[i]
            # relation_targets=torch.cat(relation_targets,0).to(self.device)
            detrectron_data.append(meta)

        loss_dict = {}
        # get detector's result
        image_det = []
        for i in image:
            image_det.append({"image":i})




        samples["sentance_tockens"] = []
        samples["relation_index"] = []
        samples["negative_text_sets"] = []
        interactive_boxes = []
        if self.task_mode == "sgdet":  # predict object
            images=torch.stack(image, 0)
            text_inputs = ['object detection [SEP]' for i in range(images.shape[0])]

            tokenized_texts = self.tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            with EventStorage() as storage:
                loss_dict,vit_output=self.detector(detrectron_data)




        # del samples, image_embeds, instance_gts
        if is_train:

            if self.predict_mode == "contrastive":
                contrastive_losses = sum(contrastive_losses) / len(samples["image"])
                loss_dict["contrastive_loss"] = contrastive_losses
                # del   neg_encoder_output,visual_relation_feature,visual_feature
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # relation_losses = sum(relation_losses) / len(samples["image"])
                # loss_dict["relation_loss"] = relation_losses
                loss_dict["loss"] = sum(loss_dict.values())
            return loss_dict

        else:
            predictions,_=self.detector(detrectron_data)
            if self.task_mode == "predcls":
                return {"predictions": predictions, "instance_gts":instance_gts}
            else:
                return {"predictions": predictions, "instance_gts": det_gts} #det_gts是大尺寸gt,和predctions对应,sgdet任务需要对iou评估，所以采用det_gt,且det_gtHe ins_gt的relation部分一致

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        # -------for initialing detectron detector ---------
        detector=model_zoo.get_config(cfg.detector_config_path).model
        # image_encoder = VisionTransformerEncoder.from_config(cfg)
        detector=instantiate(detector)

        # -------for grounding modal -----------
        # CONFIG_PATH = get_abs_path(cfg.detector_config_path)
        # grounding_model = load_model(CONFIG_PATH)

        # text encoder + multimodal encoder
        # text_encoder = XBertEncoder.from_config(cfg)
        decoder_cfg=BertConfig.from_json_file(get_abs_path(cfg.med_config_path))
        decoder_cfg.vocab_size=51
        # decoder = BertForMaskedLM(decoder_cfg)
        unified_encoder = UnifiedBertForMaskedLM.from_config(cfg)
        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 30)
        mode = cfg.get("predict_mode", "bert")
        task_mode = cfg.get("task_mode")
        window_size = cfg.get("window_size")
        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )
        RelationSampler = RelationSampling(
            cfg.RPN.POSITIVE_FRACTION,
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

        model = cls(
            detector=detector,#None, #detector, #grounding_model,
            unified_encoder=unified_encoder,
            num_classes = num_classes,
            fpn_config=fpn_config,
            proposal_generator = proposal_generator,
            roi_heads = roi_heads,
            RelationSampler = RelationSampler,
            alpha=alpha,
            momentum=momentum,
            max_txt_len=max_txt_len,
            mode=mode,
            task_mode = task_mode,
            window_size = window_size
        )

        model.load_checkpoint_from_config(cfg)

        return model
'''
creator: cxg
for converting pysgg structure "boxlist" to detrectron's "instances"
'''
def conver_boxlist_to_instance(boxlist):
    structure=Instances(boxlist.size[::-1]) #因为boxlist是(w,h), Instance是(h,w)
    assert boxlist.mode == "xyxy", "mode is not xyxy, so the folllowing convert is wrong"
    gt_boxes=Boxes(boxlist.bbox)

    structure.set("gt_boxes",gt_boxes)

    for field in boxlist.fields():
        structure.set(field,boxlist.get_field(field))
    structure.set("gt_classes", boxlist.get_field("labels"))
    structure.set("relation_tuple", boxlist.get_field("relation_tuple"))
    return structure


