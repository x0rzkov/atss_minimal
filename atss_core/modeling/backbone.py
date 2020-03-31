from collections import OrderedDict
from torch import nn

from atss_core.modeling.make_layers import conv_with_kaiming_uniform
from atss_core.modeling import fpn as fpn_module, resnet


def build_resnet_fpn_p3p7_backbone():
    body = resnet.ResNet()
    fpn = fpn_module.FPN(in_channels_list=[0, 512, 1024, 2048],
                         out_channels=256,
                         conv_block=conv_with_kaiming_uniform(use_gn=False, use_relu=False),
                         top_blocks=fpn_module.LastLevelP6P7(256, 256))
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = 256
    return model
