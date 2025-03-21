import json
import os
import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import BertConfig

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
def find_matches(dict1, dict2):
    matches = {}
    for key, value in dict1.items():
        if key.split()[0] in dict2:
            matches[value] = dict2[key.split()[0]]
    return matches

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
        mode="bert"# or "linear","contrastive"
    ):
        super().__init__()
        # BERTCONFIG = "configs/models/sggp_text_bert.json"  # todo 从配置文件中读入
        # BERTCONFIG = BertConfig.from_json_file(get_abs_path(BERTCONFIG))
        self.tokenizer = self.init_tokenizer()

        self.max_txt_len = max_txt_len
        self.use_distill = use_distill
        self.detector = detector
        self.mode = mode
        self.unified_encoder = unified_encoder
        cfg_file=OmegaConf.load("lavis/configs/datasets/vg/defaults_sgg.yaml")
        vocab_file=json.load(open(os.path.join("cache",cfg_file.datasets.visual_genome.build_info.annotations.train.storage[1])))
        self.idx_to_label =vocab_file['idx_to_label']
        self.predicate_to_idx =vocab_file['predicate_to_idx']
        self.predicate_to_idx['background']=0
        self.map_predicate2bert_idx = find_matches(self.predicate_to_idx, self.tokenizer.vocab)
        hidden_size = unified_encoder.config.hidden_size
        # self.text_encoder.encoder.gradient_checkpointing=True
        # self.decoder.bert.encoder.gradient_checkpointing = True
        if mode =="bert":
            # self.bert_cls = BertOnlyMLMHead(BERTCONFIG)
            self.loss_bert = CrossEntropyLoss()  # -100 index = not masked tocken
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            )
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
        # with EventStorage() as storage:
        #     loss_dict,vit_output=self.detector(detrectron_data)
        #     del  detrectron_data
        samples["sentance_tockens"] = []
        samples["relation_index"] = []
        for instance_gt in instance_gts:
            sentance_tockens = []
            label=instance_gt.get("labels")
            for triplet in instance_gt.get("relation_tuple"):
                temp='relation prediction [SEP] '+self.idx_to_label[str(int(label[int(triplet[0])]))] + ' '+'[MASK]'+' '+ self.idx_to_label[str(int(label[int(triplet[1])]))]
                sentance_tockens.append(temp)

            sentence = self.tokenizer(
                sentance_tockens,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            try:
                relation_index = torch.where(sentence["input_ids"][0] == 103)[0]  # relation_index是relation词出现的位置 #todo 考虑batch中可能relation出现位置不一致,这先按照一致计算

            except ValueError:
                print(f"The word '{temp}' is not found in the sentence.")
            samples["sentance_tockens"].append(sentence)
            # samples["relation_index"].append(relation_index)

        relation_losses = []
        predictions = []
        for image,tokenized_texts,batch_relation_targets in zip(samples['image'],samples["sentance_tockens"],relation_targets):
            encoder_output = self.unified_encoder(  # BertModel
                input_images=image.unsqueeze(0).repeat(len(tokenized_texts['input_ids']),1,1,1),#todo 目前把图像重复n次，n为一张图relation数目，有没有更好的方法？
                input_text_ids=tokenized_texts.input_ids,
                text_attention_mask=tokenized_texts.attention_mask,
                mode="multimodal",
            )
            prediction=[]
            if is_train:


                relation_loss = self.loss_bert(
                    encoder_output.logits[:,relation_index,:].view(-1, self.unified_encoder.config.vocab_size),
                    torch.tensor([self.map_predicate2bert_idx[int(item)] for item in batch_relation_targets.view(-1)],dtype=torch.long).cuda())

                relation_losses.append(relation_loss)

            if is_train is False:
                prediction = torch.cat(prediction, 0).squeeze().cpu()
                predictions.append(prediction)
            # relation_loss.append(F.cross_entropy(predictions, relation_target))


        # del samples, image_embeds, instance_gts
        if is_train:
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
        text_encoder = XBertEncoder.from_config(cfg)
        decoder_cfg=BertConfig.from_json_file(get_abs_path(cfg.med_config_path))
        decoder_cfg.vocab_size=51
        decoder = BertForMaskedLM(decoder_cfg)
        unified_encoder = UnifiedBertForMaskedLM.from_config(cfg, from_pretrained=True)
        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 30)

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
        )

        # model.load_checkpoint_from_config(cfg)

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


