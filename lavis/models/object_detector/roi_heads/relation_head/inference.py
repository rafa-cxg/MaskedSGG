# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from pysgg.config import cfg
from pysgg.structures.bounding_box import BoxList
from .utils_relation import obj_prediction_nms
from detectron2.structures import Instances, Boxes
import ipdb
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            attribute_on,
            use_gt_box=False,
            later_nms_pred_thres=0.3,

    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres

        
        self.rel_prop_on = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        self.BCE_loss = cfg.MODEL.ROI_RELATION_HEAD.USE_BINARY_LOSS

        self.use_relness_ranking = False
        if self.rel_prop_type == "rel_pn" and self.rel_prop_on:
            self.use_relness_ranking = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_RELATEDNESS_FOR_PREDICTION_RANKING
        try:
            self.map_predicate2bert_idx = torch.load(os.path.abspath('map_predicate2bert_idx.pth'))
        except:
            print("It is not zero-shot evaluation!")

    def forward(self, x, rel_pair_idxs, boxes,zeroshot=False):
        """
        re-NMS on refined object classifcations logits
        and ranking the relationship prediction according to the object and relationship
        classification scores

        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits,two_stage_pred_rel_logits = x


        rel_binarys_matrix = None
        
        # if boxes[0].has_field("relness_mat"):
        #     rel_binarys_matrix = [ each.get_field("relness_mat") for each in boxes]

            
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []

        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
                relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
            if not self.BCE_loss:
                obj_class_prob = F.softmax(obj_logit, -1)
            else:
                obj_class_prob = F.sigmoid(obj_logit)

            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.boxes_per_cls, obj_logit, self.later_nms_pred_thres)#这nms什么操作看不懂，它算了新的所有prop的概率，不符合赋值-1，但却没用它
                # obj_pred = box.get_field('pred_labels')
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                boxes_num = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = Instances( box._image_size)
                pred_boxes = Boxes( box.boxes_per_cls[torch.arange(boxes_num, device=device), regressed_box_idxs])
                boxlist.set("pred_boxes", pred_boxes)


            # boxlist.set('pred_labels', obj_class)  # (#obj, )不可能有bg,不同mode只是，多个nms计算
            # boxlist.set('pred_scores', obj_scores)  # (#obj, )
            boxlist.set('scores', obj_scores)  # (#obj, )
            boxlist.set('pred_classes', obj_class)
            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            if zeroshot:#当为true时，不把relation logits第一个位置置0


                rel_class_prob = F.softmax(rel_logit, -1)
                # for i in range(30522):
                #     if i not in self.map_predicate2bert_idx.values():
                #         rel_class_prob[:, i] = 0
                rel_scores, rel_class = rel_class_prob.max(dim=1)
            else: #第一个位置是background
                rel_class_prob = F.softmax(rel_logit, -1)
                rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                rel_class = rel_class + 1


            if rel_binarys_matrix is not None:
                rel_bin_mat = rel_binarys_matrix[i]
                relness = rel_bin_mat[rel_pair_idx[:, 0], rel_pair_idx[:, 1]]

            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            if self.use_relness_ranking:
                triple_scores = rel_scores * obj_scores0 * obj_scores1 * relness

            else:
                triple_scores = rel_scores * obj_scores0 * obj_scores1#不用rel_pn的情况下，rel的可能性就以50类中最大值当作rel_scores

            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]



            if rel_binarys_matrix is not None:
                boxlist.add_field('relness', relness[sorting_idx])
            # boxlist.add_field('gt_2stage', (boxlist.get_field('gt_2stage'))[sorting_idx])
            boxlist.set('rel_pair_idxs', rel_pair_idx)  # (#rel, 2)
            boxlist.set('rel_scores', rel_class_prob)  # (#rel, #rel_class)
            boxlist.set('pred_rel_labels', rel_labels)  # (#rel, )
            # boxlist.set('freq_logits', box.get_field('freq_logits'))

            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            # Note
            # TODO Kaihua: add a new type of element,
            #  which can have different length with boxlist (similar to field, except that once
            # the boxlist has such an element, the slicing operation should be forbidden.)
            # it is not safe to add fields about relation into boxlist!
            results.append(boxlist)

        return results


def make_roi_relation_post_processor(cfg):
    attribute_on =False
    use_gt_box = False if cfg.task_mode=="sgdet" else True
    later_nms_pred_thres = cfg.RELATION.LATER_NMS_PREDICTION_THRES

    postprocessor = PostProcessor(
        attribute_on,
        use_gt_box,
        later_nms_pred_thres,
    )
    return postprocessor
