"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
import itertools
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
from random import shuffle
import math
import numpy as np
import random

import torch
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

class InteractMaskingGenerator:
    '''
    #max_num_patches for block_masking, mask_ratio for mae masking
    '''
    def __init__(
            self, input_size,patch_size, min_num_patches=4, max_num_patches=None, mask_ratio=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.patch_size = patch_size

        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.max_num_masked_patches = int(mask_ratio * self.num_patches) if max_num_patches is None else max_num_patches
        self.mask_ratio = mask_ratio
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _random_mask(self, mask, max_mask_patches):
        delta = 0 # 真正新mask的patch数目
        for attempt in range(10): #认为10次一定能产生满足要求的mask操作
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)#确定左上角点坐标
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()# num_masked本次mask前，这块区域已经被mask的数目
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def _interact_mask(self, mask, max_mask_patches,union_box):
        delta = 0  # 真正新mask的patch数目

        h =  abs(union_box[3] - union_box[1])+1
        w = abs(union_box[2] - union_box[0])+1
        x = min(union_box[2],union_box[0])
        y = min(union_box[1], union_box[3])

        num_masked = mask[x: x+w,y: y+h].sum()
        max_mask_patches = 16
        if  delta <= max_mask_patches:
            for i in range(y, y + h):# 数组的第一个维度是y，第二个才是x！
                for j in  range(x, x + w):
                    if mask[i, j] == 0:
                        mask[i, j] = 1
                        delta += 1
        return mask
    def area_mae_mask(self, mask, box):


        h = int(abs(box[3] - box[1]))
        w = int(abs(box[2] - box[0]))
        x = int(min(box[2],box[0]))
        y = int(min(box[1], box[3]))
        num_patches = h * w
        if h*w ==1: #如果区域大小只有1个patch,则100% mask
            num_masked_patches = 1
        else:
            num_masked_patches = int(self.mask_ratio * h * w)
        area_mask = np.hstack([
            np.zeros(num_patches - num_masked_patches),
            np.ones(num_masked_patches),
        ])
        np.random.shuffle(area_mask)
        try:
            mask[y:y+h,x:x+w] = area_mask.reshape((h,w))
        except:
            a=1



        return mask

    def __call__(self, detection_results,random_mask):

        masks=[]
        for idx,detection_result in enumerate(detection_results):#include batches
            mask = np.zeros(shape=self.get_shape(), dtype=np.int64)
            num_boxes = len(detection_result["boxes"])
            detection_result["boxes"] = box_cxcywh_to_xyxy(detection_result["boxes"])*self.get_shape()[0]# 将归一化的cxcxwh转换为patch为单位的xyxy

            if num_boxes == 0:
                masks.append(random_mask[idx])
            else:
                if num_boxes != 1 :
                    combine_boxes = list(itertools.combinations(detection_result["boxes"], 2))
                    # shuffle(combine_boxes)
                    # 采样合适的box组合，而不是所有的Box随机组合
                    for boxes in combine_boxes:

                        box1 = boxes[0]
                        box2 = boxes[1]
                        xmin = torch.round((min(box1[0], box2[0])))
                        xmax = torch.round((max(box1[2], box2[2])))
                        ymin = torch.round((min(box1[1], box2[1])))
                        ymax = torch.round((max(box1[3], box2[3])))
                        # 判断是否相交，不相交的时候，计算得到的是左下和右上(或者别的)。
                        # if xmin > xmax:
                        #     temp = xmin
                        #     xmin = xmax
                        #     xmax = temp
                        # if ymin > ymax:
                        #     temp = ymin
                        #     ymin = ymax
                        #     ymax = temp
                        # if (xmax - xmin) == 0:
                        #     xmax = xmin + 1
                        # if (ymax - ymin) == 0:
                        #     ymax = ymin + 1

                        union_box = (xmin, ymin, xmax, ymax)
                        mask = self.area_mae_mask(mask, union_box)
                else:
                    box = torch.round(detection_result["boxes"][0])#此时num_box=1,所以取index=0
                    if (box[2] - box[0]) == 0:
                        box[2] = box[0] + 1
                    if (box[3] - box[1]) == 0:
                        box[3] = box[1] + 1
                    mask = self.area_mae_mask(mask, (box))
                    # mask = self._interact_mask(mask, self.max_num_masked_patches, union_box) #for block interactive masking only

                masks.append(torch.as_tensor(mask,device="cuda"))
        return torch.stack(masks,0)

    '''for detectron format (e.g., VITdet)'''
    # def __call__(self, detection_results):
    #
    #     masks=[]
    #     for detection_result in detection_results[0]:#include batches
    #         mask = np.zeros(shape=self.get_shape(), dtype=np.long)
    #         num_boxes = len(detection_result['instances'])
    #
    #         if num_boxes == 0 or num_boxes == 1:
    #             return None
    #         else:
    #             combine_boxes = list(itertools.combinations(detection_result['instances'].pred_boxes[:3],2))
    #             # shuffle(combine_boxes)
    #             #采样合适的box组合，而不是所有的Box随机组合
    #             for boxes in combine_boxes:
    #
    #                 box1 = boxes[0]
    #                 box2 = boxes[1]
    #                 xmin = int((max(box1[0], box2[0]))/self.patch_size)
    #                 xmax = int((min(box1[2], box2[2]))/self.patch_size)
    #                 ymin = int((max(box1[1], box2[1]))/self.patch_size)
    #                 ymax = int((min(box1[3], box2[3]))/self.patch_size)
    #                 #判断是否相交，不相交的时候，计算得到的是左下和右上(或者别的)。
    #                 if xmin > xmax:
    #                     temp = xmin
    #                     xmin = xmax
    #                     xmax = temp
    #                 if ymin > ymax:
    #                     temp = ymin
    #                     ymin = ymax
    #                     ymax = temp
    #                 if (xmax-xmin)== 0:
    #                     xmax = xmin+1
    #                 if  (ymax-ymin) ==0:
    #                     ymax = ymin+1
    #
    #                 union_box = (xmin,ymin,xmax,ymax)
    #                 mask = self._interact_mask(mask, self.max_num_patches, union_box)
    #
    #             masks.append(torch.as_tensor(mask,device=box1.device))
    #     return torch.stack(masks,0)
class RandomMaskingGenerator:
    '''
    from MAE code
    '''
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

class BlockMaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0 # 真正新mask的patch数目
        for attempt in range(10): #认为10次一定能产生满足要求的mask操作
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)#确定左上角点坐标
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int_)
        mask_count = 0
        while mask_count < self.num_masking_patches: #要mask固定数目的才不再mask
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
