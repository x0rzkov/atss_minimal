#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from torchvision import transforms as T


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
