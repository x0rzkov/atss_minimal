import bisect
import copy

import torch.utils.data as data

from atss_core.data.grouped_batch_sampler import GroupedBatchSampler
from atss_core.data.iteration_based_batch_sampler import IterationBasedBatchSampler
from atss_core.data.coco import COCODataset
from .collate_batch import BatchCollator
from atss_core.data.build import build_transforms


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]

        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(sampler, group_ids, images_per_batch, drop_uneven=False)
    else:
        batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=False)

    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)

    return batch_sampler


def make_data_loader(cfg, is_train=True, start_iter=0):
    if is_train:
        shuffle = True
        num_iters = cfg.max_iter
    else:
        shuffle = False
        num_iters = None

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1]

    transforms = build_transforms(cfg, is_train)
    dataset = COCODataset('/home/feiyuhuahuo/Data/coco2017/val2017',
                          '/home/feiyuhuahuo/Data/coco2017/annotations/instances_val2017.json',
                          remove_images_without_annotations=True, transforms=transforms)
    exit()
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)

    batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, cfg.bs, num_iters, start_iter)
    collator = BatchCollator(cfg.size_divisibility)

    data_loader = data.DataLoader(dataset, num_workers=8, batch_sampler=batch_sampler, collate_fn=collator)

    return data_loader
