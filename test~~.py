#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# import torch
# aa = torch.load('ATSS_R_50_FPN_1x.pth')
#
# # for kkk in kk:
# #     if 'rpn' in kkk:
# #         print(kkk)
#
# from collections import OrderedDict
# bb = OrderedDict()
#
# for one_key, v in aa['model'].items():
#     if 'rpn' in one_key:
#         new_key = one_key.replace('rpn.', '')
#         bb[new_key] = v
#     else:
#         bb[one_key] = v
#
# torch.save(bb, 'res50_1x.pth')
import random

print(random.choice((800, )))
