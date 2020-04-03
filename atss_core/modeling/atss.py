import math
import torch
from torch import nn

from .inference import ATSSPostProcessor
from .loss import ATSSLossComputation

from atss_core.layers.scale import Scale
from atss_core.layers.misc import DFConv2d
from atss_core.modeling.anchor_generator import make_anchor_generator_atss


class BoxCoder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes


class ATSSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        num_anchors = len(cfg.aspect_ratios) * cfg.scales_per_octave

        cls_tower = []
        bbox_tower = []
        for i in range(4):
            if cfg.tower_dcn and i == 3:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(conv_func(in_channels,
                                       in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=True))

            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(conv_func(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * (cfg.num_classes - 1), kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)
            centerness.append(self.centerness(box_tower))

        return logits, bbox_reg, centerness


class ATSSModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.head = ATSSHead(cfg, in_channels)
        self.loss_evaluator = ATSSLossComputation(cfg, BoxCoder(cfg))
        self.box_selector_test = ATSSPostProcessor(cfg, BoxCoder(cfg))
        self.anchor_generator = make_anchor_generator_atss(cfg)

    def forward(self, images, features, targets=None):
        box_cls, box_reg, centerness = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            loss_cls, loss_box, loss_centerness = self.loss_evaluator(box_cls, box_reg, centerness, targets, anchors)
            losses = {"loss_cls": loss_cls,
                      "loss_reg": loss_box,
                      "loss_centerness": loss_centerness}
            return None, losses
        else:
            return self.box_selector_test(box_cls, box_reg, centerness, anchors), {}
