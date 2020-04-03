from torch import nn
from atss_core.modeling.backbone import build_resnet_fpn_p3p7_backbone
from atss_core.modeling.inference import ATSSPostProcessor
from atss_core.modeling.loss import ATSSLossComputation
from atss_core.modeling.atss import ATSSHead, BoxCoder
from atss_core.modeling.anchor_generator import make_anchor_generator_atss
import pdb

class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_resnet_fpn_p3p7_backbone()

        self.head = ATSSHead(cfg, self.backbone.out_channels)
        self.loss_evaluator = ATSSLossComputation(cfg, BoxCoder(cfg))
        self.box_selector_test = ATSSPostProcessor(cfg, BoxCoder(cfg))
        self.anchor_generator = make_anchor_generator_atss(cfg)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        features = self.backbone(images.tensors)

        box_cls, box_reg, centerness = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            loss_cls, loss_box, loss_centerness = self.loss_evaluator(box_cls, box_reg, centerness, targets, anchors)
            losses = {"loss_cls": loss_cls,
                      "loss_reg": loss_box,
                      "loss_centerness": loss_centerness}
            return None, losses
        else:
            return self.box_selector_test(box_cls, box_reg, centerness, anchors)
