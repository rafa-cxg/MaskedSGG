# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import os.path as path
from .RTPB.utils import load_data

from pysgg.layers import Label_Smoothing_Regression
from pysgg.modeling.matcher import Matcher
from pysgg.modeling.utils import cat
from pysgg.config import cfg


class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        use_focal_loss=False,
        focal_loss_param=None,
        weight_path=''
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        # self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

        self.BCE_loss = cfg.MODEL.ROI_RELATION_HEAD.USE_BINARY_LOSS
        loss_weight = None
        if weight_path and path.exists(weight_path):
            loss_weight = load_data(weight_path)
            loss_weight = loss_weight.cuda()

        if use_focal_loss:
            self.rel_criterion_loss = FocalLoss(**focal_loss_param)
        else:
            self.rel_criterion_loss = nn.CrossEntropyLoss(weight=loss_weight)

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits#[num_prop,151]

        relation_logits = cat(relation_logits, dim=0)#relation_logits[128,51]
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        if len(torch.nonzero(rel_labels != -1)) == 0:
            loss_relation = None

        if torch.all(torch.isfinite(relation_logits)) != True:
            print('fuck relationloss')#distributions[torch.isfinite(torch.max(input,-1)[0])]
            rel_labels = rel_labels[torch.isfinite(torch.max(relation_logits,-1)[0])]
            relation_logits=relation_logits[torch.isfinite(torch.max(relation_logits,-1)[0])]
            # notnan=torch.where(torch.isfinite(relation_logits))
            # relation_logits=relation_logits[notnan]
            # rel_labels=rel_labels[notnan[:,0]]
            # if torch.all(torch.isnan(relation_logits)) == True:
            #     print('fuck cxg2')
        loss_relation = self.rel_criterion_loss(relation_logits[rel_labels != -1],  # 交叉熵 relation_logits[128,51]
                                            rel_labels[rel_labels != -1].long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())#因为用的是gt,loss是0 【num_all_prop,151】

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class DistributionLossComputation(object):
    def __init__(self,size_average=True,mode='customed_ce'):
        self.size_average = size_average
        self.mode=mode
        self.klloss=torch.nn.KLDivLoss(reduction="batchmean",log_target=False)
        self.cosloss =torch.nn.CosineEmbeddingLoss()
        self.mseloss =nn.MSELoss()
        # self.mode='kl_loss'
    def __call__(self, input, target,distributions):
        input = cat(input, dim=0).float()
        distributions = cat(distributions, dim=0).float()
        if torch.all(torch.isfinite(input)) != True:
            print('fuck 2stageloss')
            distributions = distributions[torch.isfinite(torch.max(input,-1)[0])]
            input=input[torch.isfinite(torch.max(input,-1)[0])]



        if self.mode=='customed_ce':
            logpt = F.log_softmax((input), dim=-1)#姑且先这样，后面考虑要不要sigmoid
            loss= -(distributions > 0)*logpt
            if self.size_average: return loss.mean()
            else: return loss.sum()
        if self.mode == 'kl_loss':
            input = F.log_softmax((input), dim=-1)
            loss=self.klloss(input,distributions)
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
        if self.mode == 'cos_loss':
            label=torch.tensor([1]).repeat((input.size()[0])).cuda()
            loss=self.cosloss(input,distributions,label)
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
        if self.mode == 'mse_loss':
            label=torch.tensor([1]).repeat((input.size()[0])).cuda()
            # input = F.softmax((input), dim=-1)
            loss=self.mseloss(input,distributions)
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
class FocalLoss(nn.Module):
    def __init__(self, alpha=None,gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax((input),dim=-1)#防止nan
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()
        if all(torch.isfinite(logpt)) != True:
            print('fuck1')
        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()

class TwoStageLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        num_rel_group,
        alpha=0.75, gamma=4.0,
            logits=True
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.num_rel_group = num_rel_group
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha, gamma, logits)


        # self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        self.criterion_loss = nn.CrossEntropyLoss()

        self.BCE_loss = cfg.MODEL.ROI_RELATION_HEAD.USE_BINARY_LOSS


    def __call__(self, proposals, rel_labels, relation_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """



        relation_logits = cat(relation_logits, dim=0)#relation_logits[128,51] relation的预测结果
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)#label指的是object
        rel_labels = cat(rel_labels, dim=0)
        if len(torch.nonzero(rel_labels != -1)) == 0:
            loss_relation = None
        else:
            loss_relation = self.focal_loss(relation_logits[rel_labels != -1],#交叉熵 relation_logits[128,51]
                                                rel_labels[rel_labels != -1].long())
            # loss_relation = self.focal_loss(relation_logits[rel_labels != -1],  rel_labels[rel_labels != -1].long()).sum(-1).mean(-1)

        return loss_relation

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        use_focal_loss=cfg.MODEL.ROI_RELATION_HEAD.USE_FOCAL_LOSS,
        focal_loss_param={
            'gamma': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.GAMMA,
            'alpha': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.ALPHA,
            'size_average': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.SIZE_AVERAGE
        },
        weight_path=cfg.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_PATH,
    )

    return loss_evaluator

def make_two_stage_loss_evaluator(cfg):

    loss_evaluator = TwoStageLossComputation(
        cfg.MODEL.NUM_REL_GROUP
    )

    return loss_evaluator

def make_loss_evaluator_distribution(cfg):
    loss_evaluator = DistributionLossComputation(mode=cfg.MODEL.TWO_STAGE_HEAD.LOSS_TYPE)
    return loss_evaluator