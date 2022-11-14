# -*- coding: utf-8 -*-

import torch
from torch import nn
from math import ceil
import numpy as np
import torch.nn.functional as F

from . import BaseBlocks as bb

class EfficientNet(nn.Module):
    def __init__(self, scale_width=1, scale_depth=1, drop_rate = 0.2, output_size=1000):
        super(EfficientNet, self).__init__()
        base_widths=[
            {
                "in_dim":32,
                "out_dim":16
            },
            {
                "in_dim":16,
                "out_dim":24
            },
            {
                "in_dim":24,
                "out_dim":40
            },
            {
                "in_dim":40,
                "out_dim":80
            },
            {
                "in_dim":80,
                "out_dim":112
            },
            {
                "in_dim":112,
                "out_dim":192
            },
            {
                "in_dim":192,
                "out_dim":320
            },
            {
                "in_dim":320,
                "out_dim":1280
            }
        ]
        base_depths = [1,2,2,3,3,4,1]
        kernels = [3,3,5,3,5,5,3]
        strides = [1,2,2,2,1,2,1]
        self.drop_rate = drop_rate
        scaled_widths = []
        for i in range(len(base_widths)):
            scaled_widths.append((self.do_scale_width(base_widths[i]["in_dim"], scale_width), self.do_scale_width(base_widths[i]["out_dim"], scale_width)))
        scaled_depths = [ceil(scale_depth*d) for d in base_depths]
        drop_rates = np.linspace(self.drop_rate/sum(scaled_depths), self.drop_rate, sum(scaled_depths))
        
        self.pre = bb.ConvBlock(3, scaled_widths[0][0], kernel_size=3, stride=2, padding=1)
        mbconv_layers = []
        count=0
        for i in range(7):
            d = scaled_depths[i]
            mb_type=6
            r=24
            if i ==0:
                mb_type=1
                r=4
            lays = [bb.MBConvBlock(scaled_widths[i][0], scaled_widths[i][1], kernel_size=kernels[i], stride= strides[i], drop_prob=drop_rates[i], mb_type=mb_type, r_factor=r)]
            count+=1
            for j in range(d-1):
                lays.append(bb.MBConvBlock(scaled_widths[i][1], scaled_widths[i][1], kernel_size=kernels[i], stride= 1, drop_prob=drop_rates[count], mb_type=mb_type, r_factor=r))
                count+=1
            mbconv_layers.append(nn.Sequential(*lays))
        mbconv_layers.append(bb.ConvBlock(scaled_widths[7][0], scaled_widths[7][1], kernel_size=1, stride=1, padding=0))
        self.mbconv_layers = nn.Sequential(*mbconv_layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(),
                              nn.Linear(scaled_widths[-1][1], output_size))
    
    def do_scale_width(self, w, scale_factor):
        w *= scale_factor
        new_w = (int(w+4) // 8) * 8
        new_w = max(8, new_w)
        if new_w < 0.9*w:
           new_w += 8
        return int(new_w)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.mbconv_layers(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)