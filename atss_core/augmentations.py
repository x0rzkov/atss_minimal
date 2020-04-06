#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from torchvision import transforms as T
from atss_core.data import transforms as TT

import random
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def Resize(image, target, min_size, max_size):
    h, w, _ = image.shape

    size = min_size

    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    image = F.resize(image, size)
    if isinstance(target, list):
        target = [t.resize(image.size) for t in target]
    elif target is None:
        return image
    else:
        target = target.resize(image.size)
    return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


def detect_aug(cfg):
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    to_bgr_transform = T.Lambda(lambda x: x * 255)
    normalize_transform = T.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)

    aug = T.Compose([T.ToPILImage(),
                     T.Resize(cfg.min_img_size),
                     T.ToTensor(),
                     to_bgr_transform,
                     normalize_transform])
    return aug


def val_aug(cfg):
    aug = Compose([Resize(800, 1333),
                   ToTensor(),
                   Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std, to_bgr255=True)])
    return aug
