from torch import nn
from atss_core.structures.image_list import to_image_list
from atss_core.modeling.backbone import build_resnet_fpn_p3p7_backbone
from atss_core.modeling.rpn.atss.atss import ATSSModule
import pdb


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_resnet_fpn_p3p7_backbone()
        self.rpn = ATSSModule(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
