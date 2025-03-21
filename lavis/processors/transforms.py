# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F, transforms
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            try:
                image, target = t(image, target)
            except:
                image = t(image) #避开不能使用target作为输入的变换函数
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = int(self.min_size[0])
        # size = int(random.uniform(self.min_size[0],self.min_size[1]))#size指的就是最小值
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:#根据最小边放缩，若最长边大于max_size
                size = int(round(max_size * min_original_size / max_original_size))#更新最小边（就是size对应值）

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = round(size * h / w)
            if oh > max_size:
                oh = max_size
        else:
            oh = size
            ow = round(size * w / h)
            if ow > max_size:
                ow = max_size

        return (oh, ow)

    def __call__(self, image, target=None):
        # image = F.resize(image, (self.max_size,self.max_size))
        size = self.get_size(image.size) #image是PIL格式，size是（h,w）,F中用的是(h,w)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(size[::-1])#这里它把target size缩放好了，size格式
        return image, target

class SingleResize(object): #用于生成长宽一致的图片
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = int(self.min_size[0])
        # size = int(random.uniform(self.min_size[0],self.min_size[1]))#size指的就是最小值
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:#根据最小边放缩，若最长边大于max_size
                size = int(round(max_size * min_original_size / max_original_size))#更新最小边（就是size对应值）

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = round(size * h / w)
            if oh > max_size:
                oh = max_size
        else:
            oh = size
            ow = round(size * w / h)
            if ow > max_size:
                ow = max_size

        return (oh, ow)

    def __call__(self, image, target=None):
        image = F.resize(image, (self.max_size,self.max_size))
        # size = self.get_size(image.size) #image是PIL格式，size是（h,w）,F中用的是(h,w)
        # image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size[::-1])#这里它把target size缩放好了，size格式
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std,to_bgr255=True):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (1., 1.,1.)

        self.to_bgr255 = to_bgr255
        if self.to_bgr255:
            mean = (102.9801, 115.9465, 122.7717)
            std = (1., 1.,1.)
        self.mean = mean
        self.std = std
        self.norm = transforms.Normalize(self.mean, self.std)


    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = self.norm(image)
        if target is None:
            return image
        return image, target

