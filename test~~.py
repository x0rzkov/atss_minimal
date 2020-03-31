#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from collections import OrderedDict
metrics = {"bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"]}
results = OrderedDict()
results['bbox'] = OrderedDict([(metric, -1) for metric in metrics['bbox']])
print(results)