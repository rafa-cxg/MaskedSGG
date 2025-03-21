"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

import numpy
import torch
from torchvision.transforms import Resize

from lavis.common.masking_generator import RandomMaskingGenerator
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch import nn
import torch.nn.functional as F
import lavis.processors.transforms  as T
from lavis.processors.randaugment import RandomAugment

class SggpImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean =(0.485, 0.456, 0.406) #(0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)#(0.229, 0.224, 0.225)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("sggp_caption")
class SggpCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("sggp_question")
class SggpQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("sggp_image_train")
class SggpImagePreTrainProcessor(SggpImageBaseProcessor):
    def __init__(
            self, image_size=224, patch_size=16, window_size=14, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        # self.block_masking = BlockMaskingGenerator(window_size, num_masking_patches=75,min_num_patches=16)
        self.window_size = window_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                # transforms.RandomResizedCrop(
                #     image_size,
                #     scale=(min_scale, max_scale),
                #     interpolation=InterpolationMode.BICUBIC,
                # ),
                # transforms.RandomHorizontalFlip(),
                # RandomAugment(
                #     2,
                #     5,
                #     isPIL=True,
                #     augs=[
                #         "Identity",
                #         "AutoContrast",
                #         "Brightness",
                #         "Sharpness",
                #         "Equalize",
                #         "ShearX",
                #         "ShearY",
                #         "TranslateX",
                #         "TranslateY",
                #         "Rotate",
                #     ],
                # ),
                transforms.ToTensor(),
                self.normalize

            ]
        )
        # for normal block masking
        # self.block_mask = BlockMaskingGenerator(window_size, num_masking_patches=16, min_num_patches=1)
        self.random_mask = RandomMaskingGenerator(input_size=window_size,  mask_ratio=0.6)
    def __call__(self, item):
        return self.transform(item),  self.random_mask().reshape((self.window_size,self.window_size)) #self.block_mask()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)
        patch_size = cfg.get("patch_size")
        window_size = image_size // patch_size

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

@registry.register_processor("sggp_finetune_train")
class SggpImageTrainProcessor(SggpImageBaseProcessor):
    def __init__(
            self, min_size,max_size, small_min_size,small_max_size,
            flip_horizontal_prob,
            flip_vertical_prob,
            mean=None, std=None
    ):
        super().__init__(mean=mean, std=std)

        self.resize = T.Resize(small_min_size, small_max_size)
        self.transform =  T.Compose(
        [
            # Resize(
            #     size=(max_size, max_size)),  # todo 改为比例固定的resize
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            # RandomAugment(
            #     2,
            #     5,
            #     isPIL=True,
            #     augs=[
            #         "Identity",
            #         "AutoContrast",
            #         "Brightness",
            #         "Sharpness",
            #         "Equalize",
            #         # "ShearX",
            #         # "ShearY",
            #         # "TranslateX",
            #         # "TranslateY",
            #
            #     ]),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self.transform_small = T.Compose(
            [
                T.SingleResize(small_min_size, small_max_size),
                # T.Resize(small_min_size, small_max_size),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        # "AutoContrast",
                        # "Brightness",
                        # "Sharpness",
                        # "Equalize",
                        # "ShearX",
                        # "ShearY",
                        # "TranslateX",
                        # "TranslateY",

                    ]),
                T.ToTensor(),
                T.Normalize(mean, std,to_bgr255=False),#预训练用的是非bgr255,所以sggp保持false
            ]
    )

    def __call__(self, img, target):

        large_img,large_target = self.transform(img,target)
        small_img,small_target =self.transform_small(img,target)

        return  large_img,large_target,small_img, small_target

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        min_size = cfg.get("min_size",224)
        max_size = cfg.get("max_size",224)
        small_min_size = cfg.get("low_res_min_size",224)
        small_max_size = cfg.get("low_res_max_size",224)
        flip_horizontal_prob = cfg.get("flip_horizontal_prob",0.)
        flip_vertical_prob = cfg.get("flip_vertical_prob",0.)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)



        return cls(
            min_size,
            max_size,
            small_min_size,
            small_max_size,
            flip_horizontal_prob,
            flip_vertical_prob,
            mean=mean,
            std=std,
        )


@registry.register_processor("sggp_finetune_eval")
class SggpImageEvalProcessor(SggpImageBaseProcessor):
    def __init__(
            self, min_size,max_size,small_min_size,small_max_size,
            mean=None, std=None
    ):
        super().__init__(mean=mean, std=std)
        self.resize = T.Resize(small_min_size, small_max_size)


        self.transform =  T.Compose(
        [

            T.Resize(min_size, max_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
        self.transform_small = T.Compose(
            [
                T.SingleResize(small_min_size, small_max_size),
                # T.Resize(small_min_size, small_max_size),

                T.ToTensor(),
                T.Normalize(mean, std,to_bgr255=False),
            ]
        )

    def __call__(self, img, target):
        large_img, large_target = self.transform(img, target)
        small_img, small_target =self.transform_small(img, target)

        return large_img, large_target, small_img, small_target

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        min_size = cfg.get("min_size",1024)
        max_size = cfg.get("max_size",1024)
        small_min_size = cfg.get("low_res_min_size", 224)
        small_max_size = cfg.get("low_res_max_size", 224)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)



        return cls(
            min_size,
            max_size,
            small_min_size,
            small_max_size,
            mean=mean,
            std=std,
        )


class RelationSampling(object):
    def __init__(
            self,
            fg_thres,
            require_overlap,
            num_sample_per_gt_rel,
            batch_size_per_image,  # 1000s
            positive_fraction,
            max_proposal_pairs,
            use_gt_box,
            test_overlap,
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image  # 特指relation预测的batchsize
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.max_proposal_pairs = max_proposal_pairs
        self.test_overlap = test_overlap

    def prepare_test_pairs(self, device, proposals):
        # prepare object pairs for relation prediction
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > self.max_proposal_pairs:  # 判断最多relation配对数目和max大小比较
                try:
                    pairs_qualities = p.scores
                    pairs_qualities = pairs_qualities[idxs[:, 0]] * pairs_qualities[idxs[:, 1]]
                    select_idx = torch.sort(pairs_qualities, descending=True)[-1][: self.max_proposal_pairs]
                    idxs = idxs[select_idx]
                except:
                    select_idx = torch.randint(0, len(idxs), (self.max_proposal_pairs,))
                    idxs = idxs[select_idx]
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    '''就是用于已知gt box作为proposal的采样，最终获得固定数目的pair,仅仅是把fg bg的rel总数限制在1000而已（采样background）'''

    def gtbox_relsample(self, proposals, targets):  # proposals、targets数目、label、坐标都一致
        assert self.use_gt_box
        '''num_pos_per_img是指的relation pair个数'''
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get("relation")  # [tgt, tgt]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

            # because we use the all gt boxes, so the location matching is all ones
            locating_match_stat = torch.ones((len(proposal)), device=device)  # 表明哪些prop和gt box对应
            proposal.add_field("locating_match", locating_match_stat)

            # sym_binary_rels 对称
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)  # 没有relation的proposal pair

            # generate fg bg rel_pairs 说明relation预测也是分fg bg的relpair
            if tgt_pair_idxs.shape[0] > num_pos_per_img:  # 250
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]  # 取了特定数量的背景relation对

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)  # 把fg bg relation对拼在一起作训练input
            img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long()),
                                       # img_rel_label:固定数目的fg\bg relation对的label(250)
                                       dim=0).contiguous().view(-1)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)  # 1000*0.25
        rel_idx_pairs = []
        rel_labels = []
        rel_labels_all = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.pred_classes.device
            prp_box = proposal.pred_boxes  # prp即proposal
            prp_lab = proposal.pred_classes.long()
            tgt_box = target.gt_boxes
            tgt_lab = target.labels.long()
            tgt_rel_matrix = target.relation  # [tgt, tgt]
            from detectron2.structures import pairwise_iou
            # IoU matching for object detection results
            ious = pairwise_iou(tgt_box, prp_box)  # [tgt, prp]torch.Size([20, 80])
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (
                        ious > self.fg_thres)  # [tgt, prp] self.fg_thres=0.5即iou大于0,5且类别预测对应上的
            # one box may match multiple gt boxes here we just mark them as a valid matching if they
            # match any boxes
            locating_match = (ious > self.fg_thres).nonzero()  # [tgt, prp]
            locating_match_stat = torch.zeros((len(proposal)), device=device)
            locating_match_stat[locating_match[:, 1]] = 1  # 有gt匹配的prop设置为1（根据fg_thres）
            proposal.set("locating_match", locating_match_stat)

            # Proposal self IoU to filter non-overlap
            prp_self_iou = pairwise_iou(prp_box, prp_box)  # [prp, prp]proposal之间重叠过高的
            if self.require_overlap and (not self.use_gt_box):  # default false
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = len(prp_box)  # 80
                # [prp, prp] mark the affinity relation between the det prediction
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() \
                                  - torch.eye(num_prp, device=device).long()  # 消除对角线
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0  # 对第一个维度筛选
            rel_possibility[:, prp_lab == 0] = 0  # 对第二个维度筛选

            img_rel_triplets, corrsp_gt_rel_idx, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix,
                                                                                            ious, is_match,
                                                                                            rel_possibility,
                                                                                            proposal.scores
                                                                                                )

            if target.has("relation_non_masked"):
                rel_map = target.get_field("relation_non_masked")
                gt_rel_idx = torch.nonzero(rel_map != 0)
                fg_gt_rel_pair_idx = gt_rel_idx[corrsp_gt_rel_idx[corrsp_gt_rel_idx >= 0]]
                bg_size = len(corrsp_gt_rel_idx) - len(torch.nonzero(corrsp_gt_rel_idx >= 0))
                fg_labels = rel_map[fg_gt_rel_pair_idx[:, 0].contiguous().view(-1),
                                    fg_gt_rel_pair_idx[:, 1].contiguous().view(-1)].long()
                bg_labels = torch.zeros((bg_size), device=device, dtype=torch.long)
                rel_labels_all.append(torch.cat((fg_labels, bg_labels), dim=0))

            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        if len(rel_labels_all) == 0:
            rel_labels_all = rel_labels

        return proposals, rel_labels, rel_labels_all, rel_idx_pairs, rel_sym_binarys

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility, proposals_quality):
        """
        prepare to sample fg relation triplet and bg relation triplet
        the motifs sampling method only sampled the relation pairs whose boxes are overlapping with the
        ground truth

        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]

        return:
            the sampled relation labels [num_rel_proposal, 3]
            binary_relatedness: the box pairs with that match the ground truth
                                [num_prp, num_prp]

        """

        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix != 0)  # gt box互相有rel的索引

        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)  # subject
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)  # object
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]  # gt relation数目
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)80个proposal是head_idx的mask
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp tail)
        binary_rel_matrixs = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []

        corrsp_gt_rel_idx = []

        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel_matrixs[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel_matrixs[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)

            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs from the boxes pairs matching with the ground truth
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            '''#####################################################################################'''
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device) \
                .view(-1, 1)

            fg_rel_i = torch.cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(
                torch.int64)  # 一个triblet,sub_proposal编号，obj的,relation_label
            # select higher quality proposals as fg if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = numpy.random.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

            corrsp_gt_rel_idx.extend([i, ] * fg_rel_i.shape[0])

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = torch.cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = torch.cat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)
        # we make sure that positive and negative samples in fixed ratio
        # num_pos_per_img = 5 if self.num_pos_per_img < 5 else self.num_pos_per_img
        # num_neg_per_img = (1 - self.positive_fraction) / self.positive_fraction * num_pos_per_img

        # or take as much samples as we can
        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            # samples from the pairs grouped by the high quality proposals
            pairs_qualities = proposals_quality[bg_rel_triplets[:, 0]] * proposals_quality[bg_rel_triplets[:, 1]]
            _, sorted_idx = torch.sort(pairs_qualities, dim=0, descending=True)
            bg_rel_triplets = bg_rel_triplets[sorted_idx][: int(num_neg_per_img * 2.0)]
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            _, idx = torch.sort(proposals_quality, descending=True)
            bg_rel_triplets = torch.zeros((2, 3), dtype=torch.int64, device=device)
            for i in range(2):
                bg_rel_triplets[i, 0] = 0
                bg_rel_triplets[i, 1] = 0
                bg_rel_triplets[i, 2] = 0

        corrsp_gt_rel_idx.extend([-1, ] * bg_rel_triplets.shape[0])
        corrsp_gt_rel_idx = torch.Tensor(corrsp_gt_rel_idx).long().to(device)

        return torch.cat((fg_rel_triplets, bg_rel_triplets), dim=0), corrsp_gt_rel_idx, binary_rel_matrixs
# class PostProcessor(nn.Module):
#     """
#     From a set of classification scores, box regression and proposals,
#     computes the post-processed boxes, and applies NMS to obtain the
#     final results
#     """
#
#     def __init__(
#             self,
#
#             mode = "predcls"
#     ):
#         """
#         Arguments:
#
#         """
#         super(PostProcessor, self).__init__()
#
#
#         self.mode = mode
#
#
#
#     def forward(self, rel_logits, instance):
#         """
#         re-NMS on refined object classifcations logits
#         and ranking the relationship prediction according to the object and relationship
#         classification scores
#
#         Arguments:
#             x (tuple[tensor, tensor]): x contains the relation logits
#                 and finetuned object logits from the relation model.
#             rel_pair_idxs （list[tensor]): subject and object indice of each relation,
#                 the size of tensor is (num_rel, 2)
#             boxes (list[BoxList]): bounding boxes that are used as
#                 reference, one for ech image
#
#         Returns:
#             results (list[BoxList]): one BoxList for each image, containing
#                 the extra fields labels and scores
#         """
#
#
#         rel_binarys_matrix = None
#         rel_pair_idxs = instance.get('relation_tuple')[:, :2]
#         rel_logits = rel_logits.cpu()
#
#
#         results = []
#
#         if self.mode != "predcls":
#             obj_logit = instance.get("obj_logit")
#             obj_class_prob = F.softmax(obj_logit, -1)
#
#
#             obj_class_prob[:, 0] = 0  # set background score to 0
#             num_obj_bbox = obj_class_prob.shape[0]
#             num_obj_class = obj_class_prob.shape[1]
#
#             # NOTE: by kaihua, apply late nms for object prediction
#             obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
#             obj_pred = obj_pred + 1
#             # obj_pred = box.get_field('pred_labels')
#             obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
#             obj_scores = obj_class_prob.view(-1)[obj_score_ind]
#
#             assert obj_scores.shape[0] == num_obj_bbox
#
#
#
#             # sorting triples according to score production
#             obj_scores0 = obj_scores[rel_pair_idxs[:, 0]]
#             obj_scores1 = obj_scores[rel_pair_idxs[:, 1]]
#             rel_class_prob = F.softmax(rel_logits, -1)
#             rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
#             rel_class = rel_class + 1
#
#         else:
#             rel_class_prob = F.softmax(rel_logits, -1)
#             rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
#             rel_class = rel_class + 1
#             obj_scores0=obj_scores1=torch.ones((instance.get('relation_tuple').shape[0], 1)).squeeze()
#         # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
#
#
#         triple_scores = rel_scores * obj_scores0 * obj_scores1  # 不用rel_pn的情况下，rel的可能性就以50类中最大值当作rel_scores
#
#         _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
#         rel_pair_idxs = rel_pair_idxs[sorting_idx]
#         rel_class_prob = rel_class_prob[sorting_idx]
#         rel_class = rel_class[sorting_idx]
#
#         instance.set('relation_tuple',torch.cat((rel_pair_idxs,instance.get('relation_tuple')[:, 2][sorting_idx].unsqueeze(-1)),-1))
#         rel_logits = rel_class_prob
#
#         # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
#         # Note
#         # TODO Kaihua: add a new type of element,
#         #  which can have different length with boxlist (similar to field, except that once
#         # the boxlist has such an element, the slicing operation should be forbidden.)
#         # it is not safe to add fields about relation into boxlist!
#
#
#         return rel_logits

