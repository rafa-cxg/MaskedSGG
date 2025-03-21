import copy
import itertools
import json
import os
import warnings
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F
from math import floor, ceil
from omegaconf import OmegaConf
from transformers import BertConfig
from transformers.activations import ACT2FN

from detectron2 import model_zoo
from detectron2.config import instantiate
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
from detectron2.structures import Instances,Boxes
from detectron2.utils.events import EventStorage
from lavis.common.utils import get_abs_path
from torch.nn import CrossEntropyLoss
# from lavis.datasets.datasets.vg_sgg_datasets import BoxList
from lavis.processors.sggp_processors import PostProcessor

from lavis.common.masking_generator import InteractMaskingGenerator
from detectron2.layers.roi_align import ROIAlign
import math

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
        momentum=0.995,
        alpha=0.4,
        use_distill=True,
        max_txt_len=30,
        mode="bert"# or "zero-shot" "contrastive"
    ):
        super().__init__()
        # BERTCONFIG = "configs/models/sggp_text_bert.json"  # todo 从配置文件中读入
        # BERTCONFIG = BertConfig.from_json_file(get_abs_path(BERTCONFIG))
        self.tokenizer = self.init_tokenizer()

        self.max_txt_len = max_txt_len
        self.use_distill = use_distill
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

        self.tesk = "predcls"
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



    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):

        # construct detrectron2 data structure:
        instance_gts=[]
        for label in samples["labels"]:
            instance_gts.append(conver_boxlist_to_instance(label))
        detrectron_data=[]
        relation_targets=[]

        # strack image list to bs tensor
        # image=torch.stack(samples["image"], 0)
        # image_embeds = self.detector.backbone.net(image)['last_feat']
        for i in range(len(samples["index"])):

            relation_targets.append(instance_gts[i].get("relation_tuple")[:, -1].to(self.device))

            meta = {}
            for key, value in samples.items():
                if isinstance(value,int):
                    continue
                meta[key]=value[i]
                meta["instances"]=instance_gts[i]
            # relation_targets=torch.cat(relation_targets,0).to(self.device)
            detrectron_data.append(meta)

        loss_dict = {}
        # get detector's result
        image_det = []

        # with EventStorage() as storage:
        #     loss_dict,vit_output=self.detector(detrectron_data)
        #     del  detrectron_data
        # with torch.no_grad():
        #     for i in samples["image"]:
        #         image_det.append({"image": i})
        #     self.detector.eval()
        #     detection_results = self.detector(image_det)


        samples["sentance_tockens"] = []
        samples["relation_index"] = []
        samples["negative_text_sets"] = []
        interactive_boxes = []
        for instance_gt in instance_gts:
            sentance_tockens = []
            negative_text_sets = []
            interactive_box = []
            label=instance_gt.get("labels")
            for triplet in instance_gt.get("relation_tuple"):

                if self.predict_mode == "contrastive":
                    # 准备text prompt
                    if is_train:
                        get_feature_text_inputs = "get visual interactive feature [SEP] "
                        text_inputs = " relation between {} and {} is {} ".format(
                                       self.idx_to_label[str(int(label[int(triplet[0])]))],
                                           self.idx_to_label[str(int(label[int(triplet[1])]))],self.idx_to_predicate[str(int(triplet[2]))])
                        # text_inputs = "the relation between two objects are {} ".format(self.idx_to_predicate[str(int(triplet[2]))])

                        negative_text_set = [f"the relation between {self.idx_to_label[str(int(label[int(triplet[0])]))]} and {self.idx_to_label[str(int(label[int(triplet[1])]))]} is {c} " for c in list(self.predicate_to_idx.keys()) if c != self.idx_to_predicate[str(int(triplet[2]))] ]
                        negative_text_set = random.sample(negative_text_set, 2)
                        sentance_tockens.append(text_inputs)
                        negative_text_sets.append(negative_text_set)
                    else:
                        get_feature_text_inputs = "get visual interactive feature [SEP] "
                        text_inputs = [f" relation between {self.idx_to_label[str(int(label[int(triplet[0])]))]} and " \
                                       f"{ self.idx_to_label[str(int(label[int(triplet[1])]))]} are {r} " for r in
                                      self.predicate_to_idx.keys()]
                        # text_inputs = [f"the relation between two objects are {c} " for c in self.predicate_to_idx.keys()]
                        # text_inputs = " relation between {} and {} is [MASK] ".format(
                        #     self.idx_to_label[str(int(label[int(triplet[0])]))],
                        #     self.idx_to_label[str(int(label[int(triplet[1])]))])
                        sentance_tockens.append(text_inputs)
                        # sentance_tockens = text_inputs


                else:
                    text_inputs = 'relation prediction [SEP]  ' + self.idx_to_label[
                        str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                                      str(int(label[int(triplet[1])]))]
                    # temp = 'the scene of  :' + self.idx_to_label[
                    #     str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                    #            str(int(label[int(triplet[1])]))]
                    # temp =  self.idx_to_label[
                    #     str(int(label[int(triplet[0])]))] + ' ' + '[MASK]' + ' ' + self.idx_to_label[
                    #            str(int(label[int(triplet[1])]))]
                    sentance_tockens.append(text_inputs)
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
            if self.predict_mode == "contrastive":
                interactive_boxes.append(torch.stack(interactive_box))
                # samples["negative_text_sets"].append(negative_text_sets)
                # for c in self.predicate_to_idx.keys():
                #     sentance_tockens.append("the relation between two objects are {} ".format(c) )
            # interactive_boxes.append(interactive_box)

            samples["sentance_tockens"].append(sentance_tockens)
            # samples["relation_index"].append(relation_index)

        relation_losses = []
        contrastive_losses = []
        predictions = []

        logits_per_image = []
        for i,(image,sentance_tocken,batch_relation_targets,instance_gt) in enumerate(zip(samples['image'], samples["sentance_tockens"],relation_targets,instance_gts)):
            prediction = []
            if  (self.predict_mode != "contrastive" or is_train) == True:
                tokenized_texts = self.tokenizer(
                    sentance_tocken,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

                try:
                    relation_index = torch.where(tokenized_texts["input_ids"] == 103)[1]  # relation_index是relation词出现的位置 #todo 考虑batch中可能relation出现位置不一致,这先按照一致计算
                    if len(relation_index)==0:
                        relation_index = 8
                except:
                    relation_index = 8

            if  self.predict_mode != "contrastive":
                encoder_output = self.unified_encoder(  # BertModel
                input_images=image.unsqueeze(0).repeat(len(tokenized_texts['input_ids']),1,1,1),#todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
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

                if is_train is False:
                    predictions.append(encoder_output.logits[:, relation_index, :].squeeze(-2).cpu())

            if self.predict_mode == "bert":
                pred = []
                # 用于更换unify encoder里的bert_cls情况下的训练，sggp_unified_bert.json 的decoder_vocab_size设置为51
                for i, index in enumerate(relation_index):
                    pred.append(encoder_output.logits[i, index, :].unsqueeze(0))
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
                    predictions.append(pred)

            if self.predict_mode =="contrastive":

                interactive_box_perimg = interactive_boxes[i]
                # 1为匹配，0为负样本
                # expanded_tensor = batch_relation_targets.unsqueeze(1)
                # pos_neg_matrix = (batch_relation_targets == expanded_tensor).to(batch_relation_targets.device)

                # visual_prompt = ["interction of two regions " for _ in range(interactive_box.shape[0])]
                # tockenize_visual_text = self.tokenizer(
                #     visual_prompt,
                #     padding="max_length",
                #     truncation=True,
                #     max_length=self.max_txt_len,
                #     return_tensors="pt",
                # ).to(self.device)
                # visual_output = self.unified_encoder(  # BertModel
                #     input_images=image.unsqueeze(0).repeat(interactive_box.shape[0], 1, 1, 1),
                #     # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                #     input_text_ids=tockenize_visual_text.input_ids,
                #     text_attention_mask=tockenize_visual_text.attention_mask,
                #     mode=self.predict_mode)

                tokenized_get_visual_texts = self.tokenizer(
                    get_feature_text_inputs,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

                # get visual feature
                visual_encoder_output = self.unified_encoder(  # BertModel
                    input_images=image.unsqueeze(0).repeat(len(tokenized_get_visual_texts['input_ids']), 1, 1, 1),
                    # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                    input_text_ids=tokenized_get_visual_texts.input_ids,
                    text_attention_mask=tokenized_get_visual_texts.attention_mask,
                    mode=self.predict_mode)


                # patch pooling
                interactive_box = torch.cat((torch.tensor(0).unsqueeze(0).unsqueeze(1).expand(interactive_box_perimg.shape[0], -1)
                                             , interactive_box_perimg), 1)

                visual_feature = ROIAlign((1, 1), 1, 0, aligned=True).forward(
                    visual_encoder_output.hidden_states[:, 1:197, :].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                    interactive_box.cuda()).squeeze(1)
                visual_feature = self.vision_proj(
                    visual_feature.reshape(visual_feature.shape[0], -1))
                if is_train:
                    #prepare contrastive samples
                    pos_encoder_output = self.unified_encoder(  # BertModel
                        input_images=image.unsqueeze(0).repeat(len(tokenized_texts['input_ids']), 1, 1, 1),
                        # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                        input_text_ids=tokenized_texts.input_ids,
                        text_attention_mask=tokenized_texts.attention_mask,
                        mode=self.predict_mode)
                    # patch pooling
                    visual_pos_relation_feature = ROIAlign((1, 1), 1, 0, aligned=True).forward(
                        pos_encoder_output.hidden_states[:, 1:197, :].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                        interactive_box.cuda()).squeeze(1)
                    visual_pos_relation_feature = self.vision_proj(
                        visual_pos_relation_feature.reshape(visual_pos_relation_feature.shape[0], -1))
                    contrastive_loss = 0
                    for neg_sample in negative_text_sets:
                        tokenized_neg_visual_texts = self.tokenizer(
                            neg_sample,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(self.device)
                        neg_encoder_output = self.unified_encoder(  # BertModel
                            input_images=image.unsqueeze(0).repeat(len(tokenized_neg_visual_texts['input_ids']), 1, 1, 1),
                            # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                            input_text_ids=tokenized_neg_visual_texts.input_ids,
                            text_attention_mask=tokenized_neg_visual_texts.attention_mask,
                            mode=self.predict_mode)
                        # patch pooling
                        interactive_box = torch.cat((torch.arange(0, len(tokenized_neg_visual_texts['input_ids'])).unsqueeze(1), interactive_box_perimg[i].unsqueeze(0).expand(len(tokenized_neg_visual_texts['input_ids']),-1)), 1)
                        visual_neg_relation_feature = ROIAlign((1, 1), 1, 0, aligned=True).forward(
                            neg_encoder_output.hidden_states[:, 1:197, :].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                            interactive_box.cuda()).squeeze(1)

                        visual_neg_relation_feature = self.vision_proj(
                            visual_neg_relation_feature.reshape(visual_neg_relation_feature.shape[0], -1))
                        labels = torch.randint(0, visual_neg_relation_feature.shape[0], (1,)).long().to(self.device)
                        # labels = torch.zeros(1).long().to(self.device)
                        part1 = visual_neg_relation_feature[:labels]
                        part2 = visual_neg_relation_feature[labels:]
                        visual_relation_feature = torch.cat((part1, visual_pos_relation_feature[None,i], part2), dim=0)
                        # normalized features
                        # visual_relation_feature = torch.cat((visual_pos_relation_feature[None,i],visual_neg_relation_feature),0)
                        visual_relation_feature = (
                                visual_relation_feature / visual_relation_feature.norm(dim=1, keepdim=True))
                        # cosine similarity as logits
                        # logit_scale = self.logit_scale.exp()
                        logits_per_instance =  visual_feature[None,i] @ visual_relation_feature.T
                        # logits_per_image.append(logits_per_instance)
                        # torch.cat(logits_per_image,0)

                        contrastive_loss += F.cross_entropy(logits_per_instance, labels)
                    contrastive_losses.append(contrastive_loss)
                    # else:
                    #     prediction.append(logits_per_instance)
                    #     predictions.append(torch.cat(prediction,0))
                else:
                    for candidate_sentance in sentance_tocken:
                        tokenized_texts = self.tokenizer(
                            candidate_sentance,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_txt_len,
                            return_tensors="pt",
                        ).to(self.device)
                        # prepare contrastive samples
                        encoder_output = self.unified_encoder(  # BertModel
                            input_images=image.unsqueeze(0).repeat(len(tokenized_texts['input_ids']), 1, 1, 1),
                            # todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                            input_text_ids=tokenized_texts.input_ids,
                            text_attention_mask=tokenized_texts.attention_mask,
                            mode=self.predict_mode)
                        # patch pooling
                        interactive_box = torch.cat((torch.arange(0, len(
                            tokenized_texts['input_ids'])).unsqueeze(1),
                                                     interactive_box_perimg[i].unsqueeze(0).expand(
                                                         len(tokenized_texts['input_ids']), -1)), 1)
                        visual_relation_feature = ROIAlign((1, 1), 1, 0, aligned=True).forward(
                            encoder_output.hidden_states[:, 1:197, :].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
                            interactive_box.cuda()).squeeze(1)
                        visual_relation_feature = self.vision_proj(
                            visual_relation_feature.reshape(visual_relation_feature.shape[0], -1))
                        visual_relation_feature = (
                                visual_relation_feature / visual_relation_feature.norm(dim=1, keepdim=True))
                        # cosine similarity as logits
                        # logit_scale = self.logit_scale.exp()
                        logits_per_instance =  visual_feature[None, i] @ visual_relation_feature.T
                        prediction.append(logits_per_instance)
                    prediction = torch.cat(prediction,0)
                predictions.append(prediction)

        # if self.predict_mode == "contrastive" and is_train ==False:
        #     relation_targets = torch.cat(relation_targets, 0)
        #     # logit_scale = self.logit_scale.exp()
        #
        #     visual_relation_features = torch.cat(visual_relation_features, 0)
        #     text_relation_features = torch.cat(text_relation_features, 0)
        #     # fusion = torch.cat((visual_relation_features,text_relation_features),1)
        #     logits = self.cls_head(text_relation_features)
        #     logits = logits.softmax(dim=-1)
        #
        #     # for box in interactive_box:
        #     #     box = torch.cat((torch.arange(0, num_rel).unsqueeze(1).expand(
        #     #         num_rel, -1), box.unsqueeze(0).expand(num_rel,-1)), 1)
        #     #
        #     #     visual_relation_feature = ROIAlign((3, 3), 1.0, 0, aligned=True).forward(
        #     #         encoder_output.hidden_states[-1][:, 1:197, :].reshape(-1, 14, 14, 768).permute(0, 3, 1, 2),
        #     #         box.cuda()).squeeze(1)
        #     #     text_relation_feature = encoder_output.hidden_states[-1][:, 197 + relation_index,
        #     #                             :]  # (num_rel,1,hidden_size)
        #     #
        #     #     # normalized features
        #     #     visual_relation_feature = self.vision_proj(visual_relation_feature.reshape(*visual_relation_feature.shape[:-2], -1))
        #     #     visual_relation_feature = (visual_relation_feature / visual_relation_feature.norm(dim=1, keepdim=True)).squeeze(2)
        #     #     text_relation_feature = (text_relation_feature / text_relation_feature.norm(dim=1, keepdim=True))
        #     #
        #     #     # cosine similarity as logits
        #     #
        #     #     logits_per_image = torch.diag(logit_scale * visual_relation_feature @ text_relation_feature.T)
        #     #
        #     #     pred = logits_per_image.softmax(dim=-1)
        #     #     prediction.append(pred)
        #     # # pred = self.postprocessor(pred, instance_gt)  # todo 只适用于51类预测!,且只能用于predcls
        #
        #     predictions.append(logits)



                # pred = encoder_output.logits[:, relation_index, :]
        # if self.predict_mode == "contrastive" and is_train:
        #
        #
        #     relation_targets = torch.cat(relation_targets,0)
        #     logit_scale = self.logit_scale.exp()
        #
        #     visual_relation_features =  torch.cat(visual_relation_features,0)
        #     text_relation_features = torch.cat(text_relation_features, 0)
        #     # fusion = torch.cat((visual_relation_features,text_relation_features),1)
        #     logits = self.cls_head(text_relation_features)
        #     contrastive_loss = F.cross_entropy(logits, relation_targets)

            # # cosine similarity as logits
            # logit_scale = self.logit_scale.exp()
            # logits_per_rel = logit_scale * visual_relation_features @ text_relation_features.t()
            # # logits_per_rel_text = logits_per_rel.t()
            #
            #
            #
            # contrastive_loss = 0
            # for i,relation_target in enumerate(relation_targets):
            #
            #     mask = torch.where(relation_targets == relation_target, torch.tensor(1,device=self.device), torch.tensor(0,device=self.device))
            #     # num_neg = torch.nonzero(mask==0).shape[0]
            #     # padding_mask=torch.zeros((51+logits_per_rel.shape[0])).to(self.device)
            #     # padding_mask[:relation_targets.shape[0]] = mask
            #     # padding_mask[relation_target+relation_targets.shape[0]]=1#增强文本中,和正样本相同的text标注为正样本
            #     logits_per_image = torch.cat((logits_per_rel[i,i].unsqueeze(0),logits_per_rel[i, mask == 0]),0).unsqueeze(0)
            #     # logits_per_text = torch.cat((logits_per_rel_text[i, i].unsqueeze(0), logits_per_rel_text[i, padding_mask == 0]),
            #     #                              0).unsqueeze(0)
            #
            #
            #     #补全负样本
            #     labels = torch.zeros(1).long().to(self.device)
            #     contrastive_loss += F.cross_entropy(logits_per_image, labels) #+ F.cross_entropy(logits_per_text, labels))/2

            # contrastive_loss /= relation_targets.shape[0]



        # del samples, image_embeds, instance_gts
        if is_train:

            if self.predict_mode == "contrastive":
                contrastive_losses = sum(contrastive_losses) / len(samples["image"])
                loss_dict["contrastive_loss"] = contrastive_losses
            else:
                relation_losses = sum(relation_losses) / len(samples["image"])
                loss_dict["relation_loss"] = relation_losses
            return {"loss": sum(loss_dict.values())}
            # return SggpOutputWithLogits(
            #     loss= sum(loss_dict.values()),
                # intermediate_output=SggpIntermediateOutput(
                #     image_embeds=image_embeds,
                #
                # ),
            # )
        else:
            return {"predictions": predictions, "instance_gts":instance_gts}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        detector=model_zoo.get_config(cfg.detector_config_path).model
        # image_encoder = VisionTransformerEncoder.from_config(cfg)
        detector=instantiate(detector)

        # text encoder + multimodal encoder
        # text_encoder = XBertEncoder.from_config(cfg)
        decoder_cfg=BertConfig.from_json_file(get_abs_path(cfg.med_config_path))
        decoder_cfg.vocab_size=51
        # decoder = BertForMaskedLM(decoder_cfg)
        unified_encoder = UnifiedBertForMaskedLM.from_config(cfg, from_pretrained=True)
        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 30)
        mode = cfg.get("predict_mode", "bert")

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            detector=detector,
            unified_encoder=unified_encoder,
            num_classes =  num_classes,
            alpha=alpha,
            momentum=momentum,
            max_txt_len=max_txt_len,
            mode=mode
        )

        model.load_checkpoint_from_config(cfg)

        return model
'''
creator: cxg
for converting pysgg structure "boxlist" to detrectron's "instances"
'''
def conver_boxlist_to_instance(boxlist):
    structure=Instances(boxlist.size)
    gt_boxes=Boxes(boxlist.bbox)
    structure.set("gt_boxes",gt_boxes)

    for field in boxlist.fields():
        structure.set(field,boxlist.get_field(field))
    structure.set("gt_classes", boxlist.get_field("labels"))
    structure.set("relation_tuple", boxlist.get_field("relation_tuple"))
    return structure


