from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.multi_scale_range[0] == -1:
            min_size = 800
        else:
            assert len(cfg.multi_scale_range) == 2, "multi_scale_range error, (lower bound, upper bound)"
            min_size = list(range(cfg.multi_scale_range[0],
                                  cfg.multi_scale_range[1] + 1))

        max_size = 1333
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = 800
        max_size = 1333
        flip_prob = 0

    normalize_transform = T.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std, to_bgr255=True)

    transform = T.Compose([T.Resize(min_size, max_size),
                           T.RandomHorizontalFlip(flip_prob),
                           T.ToTensor(),
                           normalize_transform])
    return transform
