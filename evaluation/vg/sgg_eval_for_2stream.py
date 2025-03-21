import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
from sklearn import metrics

from detectron2.structures import pairwise_iou
from detectron2.structures import Boxes

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

def     intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

# from pysgg.config import cfg
# from evaluation.vg.vg_stage_eval_utils import (
#     boxlist_iou,
#     intersect_2d_torch_tensor,
#     dump_hit_indx_dict_to_tensor,
#     trans_cluster_label,
#     ENTITY_CLUSTER,
#     PREDICATE_CLUSTER,
# )
# from pysgg.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict

    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """

    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack(
            (
                class_scores[sub_id],
                predicate_scores,
                class_scores[ob_id],
            )
        )

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
    gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres, phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)#num_gt_trip,4096]该函数return 为true的地方，代表triblet和gt triblet完全对应.(但不涉及坐标)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]#pred_box:[4096,8]是rel_pair对的box：pred_triplet_boxes
    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],#有和pred_triblet对应的gt编号
        gt_boxes[gt_has_match],#treblet坐标
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]#与第n个gt_box_pair对应上的pred_box_pair坐标 eg:(146, 8),一个gt_pair box可对应多个pred(不考虑坐标)
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0
            )

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = pairwise_iou(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = pairwise_iou(Boxes(gt_box[None, :4]), Boxes(boxes[:, :4]))[0]
            obj_iou = pairwise_iou(Boxes(gt_box[None, 4:]), Boxes(boxes[:, 4:]))[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds.numpy()]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)
        self.type = "recall"
        self.type2 = "2stagerecall"

    def register_container(self, mode):
        self.result_dict[mode + f"_{self.type}"] = {20: [], 50: [], 100: []}
        self.result_dict[mode + f"_{self.type2}"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall"].items():
            result_str += "  R @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=Recall(Main)." % mode
        result_str += "\n"
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container["pred_rel_inds"]#inds确实是编号，因为1 image里面proposal不能超过80个，所以inds也不超过80,这个编号直接从rel_pair_idx来的

        rel_scores = local_container["rel_scores"]

        gt_rels = local_container["gt_rels"]#[n,3],(sub_id, ob_id, pred_label)
        gt_classes = local_container["gt_classes"]
        gt_boxes = local_container["gt_boxes"]

        pred_classes = local_container["pred_classes"]
        pred_boxes = local_container["pred_boxes"]#[]80,4
        obj_scores = local_container["obj_scores"]#[num_prop,]
        iou_thres = global_container["iou_thres"]

        if rel_scores.shape[-1]>51:#说明分类器是30522的，属于zero-shot
            pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1)[:,None]))  # todo 为何之前是1没问题 [4096,3],最后一个元素是rel_label,如果是predcls:[num_box!,3]

            pred_scores = rel_scores.max(1)
        else:
            pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))#[4096,3],最后一个元素是rel_label,如果是predcls:[num_box!,3]

            pred_scores = rel_scores[:, 1:].max(1)#todo 为什么排除background? 因为不存在真正意义的no relation吧

        '''gt_rels:[num_rel,3],gt_classes[num_box,],gt_boxes:[num_box,4]'''
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)#gt_triplets[sub_clss,rel_class,obj_class]

        local_container["gt_triplets"] = gt_triplets

        local_container["gt_triplet_boxes"] = gt_triplet_boxes
        # local_container["gt_2stage_triplets"] = gt_2stage_triplets

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(#pred_triplet_scores:sub\pred\obj score
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores
        )



        # Compute recall. It's most efficient to match once and then do recall after
        # pred_to_gt = _compute_pred_matches(#返回的pred_to_gt不仅仅label对应，且sub\obj box坐标处于iou阈值之间
        #     gt_triplets,
        #     pred_triplets,
        #     gt_triplet_boxes,
        #     pred_triplet_boxes,
        #     iou_thres,
        #     phrdet=mode == "phrdet",
        # )
        # only for relation distribution plot
        pred_to_gt = _compute_pred_matches(#返回的pred_to_gt不仅仅label对应，且sub\obj box坐标处于iou阈值之间
            pred_triplets,
            pred_triplets,
            pred_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )
        local_container["pred_to_gt"] = pred_to_gt



        for k in self.result_dict[mode + "_recall"]:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + "_recall"][k].append(rec_i)

        return local_container

class SGTOPRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGTOPRecall, self).__init__(result_dict)
        self.type = "toprecall"
        self.val_freq=False# False表明是eval EEM, true是验证freq bias.

    def register_container(self, mode):
        self.result_dict[mode + f"_{self.type}"] = {1: [], 5: []}


    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_toprecall"].items():
            result_str += "  R @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=TOP Recall." % mode
        result_str += "\n"
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        if self.val_freq==True:
            pred_rel_inds = local_container[
                "pred_rel_inds"]  # inds确实是编号，因为1 image里面proposal不能超过80个，所以inds也不超过80,这个编号直接从rel_pair_idx来的
            rel_scores = local_container["freq_scores"]
        else:
            pred_rel_inds = local_container["pred_2stage_rel_inds"]#inds确实是编号，因为1 image里面proposal不能超过80个，所以inds也不超过80,这个编号直接从rel_pair_idx来的
            rel_scores = local_container["2stage_rel_scores"]

        gt_rels = local_container["gt_rels"]#[n,3],(sub_id, ob_id, pred_label)
        gt_classes = local_container["gt_classes"]
        gt_boxes = local_container["gt_boxes"]

        pred_classes = local_container["pred_classes"]
        pred_boxes = local_container["pred_boxes"]#[]80,4
        obj_scores = local_container["obj_scores"]#[num_prop,]
        iou_thres = global_container["iou_thres"]

        for k in self.result_dict[mode + f"_{self.type}"].keys():
            desc_relscore_index=(rel_scores[:, 1:].argsort(-1)[:,::-1])[:,:k].reshape(-1,1)
            pred_rel_inds=np.expand_dims(pred_rel_inds,1).repeat(k,1).reshape(-1,2)
            pred_rels = np.column_stack((pred_rel_inds, 1 + desc_relscore_index))#[4096,3],最后一个元素是rel_label,如果是predcls:[num_box!,3]
            try:
                rel_scores=rel_scores.repeat(k,0)
                pred_scores = rel_scores[:,1:].max(-1)
            except:
                a=1
            # pred_scores = rel_scores[:, 1:].max(1)#todo 为什么排除background? 因为不存在真正意义的no relation吧
            gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)#gt_triplets[sub_clss,rel_class,obj_class]
            pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(#pred_triplet_scores:sub\pred\obj score
            pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores
        )
            pred_to_gt = _compute_pred_matches(#返回的pred_to_gt不仅仅label对应，且sub\obj box坐标处于iou阈值之间
                gt_triplets,
                pred_triplets,
                gt_triplet_boxes,
                pred_triplet_boxes,
                iou_thres,
                phrdet=mode == "phrdet",
            )
            topk=k*100
            match = reduce(np.union1d, pred_to_gt[:topk])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + "_toprecall"][k].append(rec_i)

        # local_container["pred_to_gt"] = pred_to_gt
        #
        # local_container["gt_triplets"] = gt_triplets
        #
        # local_container["gt_triplet_boxes"] = gt_triplet_boxes


        return local_container
"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""


class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)
        self.type = "recall_nogc"

    def register_container(self, mode):
        self.result_dict[mode + "_recall_nogc"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall_nogc"].items():
            result_str += "ngR @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=No Graph Constraint Recall(Main)." % mode
        result_str += "\n"
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container["obj_scores"]
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        pred_boxes = local_container["pred_boxes"]
        pred_classes = local_container["pred_classes"]
        gt_rels = local_container["gt_rels"]

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)#把sub obj的score乘起来
        if rel_scores.shape[-1] > 51:  # 说明分类器是30522的，属于zero-shot
            nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores  # sub obj 与每一个rel对应的score:[n,50]
            nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
            nogc_pred_rels = np.column_stack(
                (pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] )
            )
            nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1]]
        else:
            nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]#sub obj 与每一个rel对应的score:[n,50]
            nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
            nogc_pred_rels = np.column_stack(
                (pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1)
            )
            nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1] + 1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
            nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container["gt_triplets"]
        gt_triplet_boxes = local_container["gt_triplet_boxes"]
        iou_thres = global_container["iou_thres"]

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode == "phrdet",
        )

        local_container["nogc_pred_to_gt"] = nogc_pred_to_gt

        for k in self.result_dict[mode + "_recall_nogc"]:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + "_recall_nogc"][k].append(rec_i)


"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""


class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)
        self.type = "zeroshot_recall"

    def register_container(self, mode):
        self.result_dict[mode + "_zeroshot_recall"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_zeroshot_recall"].items():
            result_str += " zR @ %d: %.4f; " % (k, np.mean(v))
        result_str += " for mode=%s, type=Zero Shot Recall." % mode
        result_str += "\n"
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        zeroshot_triplets = global_container["zeroshot_triplet"]

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack(
            (gt_classes[sub_id], gt_classes[ob_id], pred_label)
        )  # num_rel, 3

        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[
            0
        ].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]

        for k in self.result_dict[mode + "_zeroshot_recall"]:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = (
                    len(self.zeroshot_idx)
                    + len(match_list)
                    - len(set(self.zeroshot_idx + match_list))
                )
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + "_zeroshot_recall"][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""


class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + "_accuracy_hit"] = {20: [], 50: [], 100: []}
        self.result_dict[mode + "_accuracy_count"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_accuracy_hit"].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + "_accuracy_count"][k])
            result_str += "  A @ %d: %.4f; " % (k, a_hit / a_count)
        result_str += " for mode=%s, type=TopK Accuracy." % mode
        result_str += "\n"
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = (#[num_prop!,]
            local_container["pred_rel_inds"][:, 0] * 1024
            + local_container["pred_rel_inds"][:, 1]
        )
        gt_pair_idx = (#[gt_num_rel,]
            local_container["gt_rels"][:, 0] * 1024 + local_container["gt_rels"][:, 1]
        )
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0#[num_box!,gt_num_rel]->[num_box!,]代表预测的relation对（仅仅是sub\obj box的编号）

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]
        gt_rels = local_container["gt_rels"]

        for k in self.result_dict[mode + "_accuracy_hit"]:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
            # for sgcls and predcls
            if mode != "sgdet":
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + "_accuracy_hit"][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + "_accuracy_count"][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""


class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel

        self.print_detail = print_detail
        if len(ind_to_predicates)<=51:
            self.rel_name_list = ind_to_predicates[1:]  # remove __background__
        else:
            self.rel_name_list = ind_to_predicates
        self.type = "mean_recall"

    def register_container(self, mode):
        # self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        # self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + "_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_mean_recall_collect"] = {
            20: [[] for i in range(self.num_rel)],
            50: [[] for i in range(self.num_rel)],
            100: [[] for i in range(self.num_rel)],
        }
        self.result_dict[mode + "_mean_recall_list"] = {20: [], 50: [], 100: []}

        self.result_dict[mode + "_2stage_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            result_str += " mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Mean Recall." % mode
        result_str += "\n"
        if self.print_detail:
            result_str += "Per-class recall@50: \n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][50]
            ):
                if r!=0.00: #for 30522 printing,we only print class that isn't 0
                    result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n"
            result_str += "Per-class recall@100: \n"
            for n, r in zip(
                self.rel_name_list, self.result_dict[mode + "_mean_recall_list"][100]
            ):
                if r != 0: #for 30522 printing,we only print class that isn't 0
                    result_str += "({}:{:.4f}) ".format(str(n), r)
            result_str += "\n\n"



        return result_str



    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container["pred_to_gt"]#list:4096。pred_to_gt指的是pred的triplet和gt triblet id的对应关系。可以一对多说明不管位置，只管label对应
        # gt_rels = local_container["gt_rels"]#(sub_id, ob_id, pred_label)
        # # only for relation distribution plot
        gt_rels = np.concatenate((local_container["pred_rel_inds"],np.argmax(local_container["rel_scores"],-1)[:,None]),-1)

        for k in self.result_dict[mode + "_mean_recall_collect"]:#k=20\50\100
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])#不管有多少rel pair,只取前k个正确的数量，除以gt的数量，所以比例不会超过1
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):#gt_rels:[n,3]
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1#todo 这个是错误？why? 0位数字代表多少个rel_pair

            for idx in range(len(match)):#match里的内容实际上是match的prop的编号
                local_label = gt_rels[int(match[idx]), 2]#对应gt rel_triblet编号的gt rel_triblet类别
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):#_mean_recall_collect就是对每个rel算的recall
                # if recall_count[n] > 0:
                    # self.result_dict[mode + "_mean_recall_collect"][k][n].append(
                    #     float(recall_hit[n] / recall_count[n])
                    # )
                    # for plot distribution
                if recall_hit[n] > 0:
                    self.result_dict[mode + "_mean_recall_collect"][k][n].append(
                        float(recall_hit[n])
                    )



    '''计算除background的rel的recall。把每个图片计算的recall(recall_collect取平均)'''
    def calculate_mean_recall(self, mode):#collect里每个rel是每个图的结果列表，recall_list是每个图取平均的结果：
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            sum_recall = 0
            if self.num_rel<=51:#说明第一个元素是background
                num_rel_no_bg = self.num_rel - 1
                for idx in range(num_rel_no_bg):  # 对于每个rel class,_mean_recall_collect[k][idx + 1]是个长度为照片总数的list

                    if len(self.result_dict[mode + "_mean_recall_collect"][k][idx + 1]) == 0:
                        tmp_recall = 0.0
                    else:
                        tmp_recall = np.mean(
                            self.result_dict[mode + "_mean_recall_collect"][k][idx + 1]
                        )
                    self.result_dict[mode + "_mean_recall_list"][k].append(tmp_recall)
                    sum_recall += tmp_recall
            else:
                num_rel_no_bg = self.num_rel
                for idx in range(num_rel_no_bg):  # 对于每个rel class,_mean_recall_collect[k][idx + 1]是个长度为照片总数的list

                    if len(self.result_dict[mode + "_mean_recall_collect"][k][idx]) == 0:
                        tmp_recall = 0.0
                    else:
                        tmp_recall = np.mean(
                            self.result_dict[mode + "_mean_recall_collect"][k][idx]
                        )
                    self.result_dict[mode + "_mean_recall_list"][k].append(tmp_recall)
                    sum_recall += tmp_recall


            self.result_dict[mode + "_mean_recall"][k] = sum_recall / float(num_rel_no_bg)

        return


class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]  # remove __background__
        self.type = "ng_mean_recall"

    def register_container(self, mode):
        self.result_dict[mode + "_ng_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + "_ng_mean_recall_collect"] = {
            20: [[] for i in range(self.num_rel)],
            50: [[] for i in range(self.num_rel)],
            100: [[] for i in range(self.num_rel)],
        }
        self.result_dict[mode + "_ng_mean_recall_list"] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_ng_mean_recall"].items():
            result_str += "ng-mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=No Graph Constraint Mean Recall." % mode
        result_str += "\n"
        # if self.print_detail:
        #     result_str += "----------------------- Details ------------------------\n"
        #     for n, r in zip(
        #         self.rel_name_list, self.result_dict[mode + "_ng_mean_recall_list"][100]
        #     ):
        #         result_str += "({}:{:.4f}) ".format(str(n), r)
        #     result_str += "\n"
        #     result_str += "--------------------------------------------------------\n"

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container["nogc_pred_to_gt"]
        gt_rels = local_container["gt_rels"]

        for k in self.result_dict[mode + "_ng_mean_recall_collect"]:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + "_ng_mean_recall_collect"][k][n].append(
                        float(recall_hit[n] / recall_count[n])
                    )

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + "_ng_mean_recall"].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + "_ng_mean_recall_collect"][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(
                        self.result_dict[mode + "_ng_mean_recall_collect"][k][idx + 1]
                    )
                self.result_dict[mode + "_ng_mean_recall_list"][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + "_ng_mean_recall"][k] = sum_recall / float(num_rel_no_bg)
        return


"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""


class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + "_accumulate_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_accumulate_recall"].items():
            result_str += " aR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Accumulate Recall." % mode
        result_str += "\n"
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + "_accumulate_recall"].items():
            self.result_dict[mode + "_accumulate_recall"][k] = float(
                self.result_dict[mode + "_recall_hit"][k][0]
            ) / float(self.result_dict[mode + "_recall_count"][k][0] + 1e-10)

        return



