# -*- coding: utf-8 -*-

import torch
from torch import nn
from math import ceil
import numpy as np
import torch.nn.functional as F

class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BaseBlock, self).__init__()
        
        padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.use_skip_connect = (in_channels == out_channels) and (stride == 1)
    
    def forward(self, x):
        x_ = x.clone()
        x_ = self.conv(x)
        x_ = self.act(self.bn(x_))
        if self.use_skip_connect:
            return x_ + x 
        else:
            return x_
    
    
class JoeNet(nn.Module):
    def __init__(self, scale_depth=1, scale_width=1, output_size=10):
        super(JoeNet, self).__init__()
        self.scale_depth = scale_depth
        self.scale_width = scale_width
        
        base_widths=[
            {
                "in_dim":16,
                "out_dim":32
            },
            {
                "in_dim":32,
                "out_dim":64
            },
            {
                "in_dim":64,
                "out_dim":128
            },
            {
                "in_dim":128,
                "out_dim":256
            }
        ]
        base_depths = [1,2,3,2]
        kernels = [3,3,3,3]
        strides = [1,1,1,1]
        scaled_widths = []
        for i in range(len(base_widths)):
            scaled_widths.append((self.do_scale_width(base_widths[i]["in_dim"], scale_width), self.do_scale_width(base_widths[i]["out_dim"], scale_width)))
        scaled_depths = [ceil(scale_depth*d) for d in base_depths]
        self.pre = BaseBlock(3, scaled_widths[0][0], kernel_size=3, stride=1, padding=1)
        conv_layers = []
        
        for i in range(len(base_widths)):
            d = scaled_depths[i]
            lays = [BaseBlock(scaled_widths[i][0], scaled_widths[i][1], kernel_size=kernels[i], stride= strides[i])]
            for j in range(d-1):
                lays.append(BaseBlock(scaled_widths[i][1], scaled_widths[i][1], kernel_size=kernels[i], stride= strides[i]))
            lays.append(nn.MaxPool2d((2,2)))
            conv_layers.append(nn.Sequential(*lays))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(),
                                  nn.Linear(scaled_widths[-1][1], 512),
                                  nn.Linear(512, 512),
                                  nn.Linear(512, output_size))
        
    def do_scale_width(self, w, scale_factor):
        w *= scale_factor
        new_w = (int(w+4) // 8) * 8
        new_w = max(8, new_w)
        if new_w < 0.9*w:
           new_w += 8
        return int(new_w)
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"scale_depth": self.scale_depth, "scale_width": self.scale_width}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def forward(self, x):
        x = self.pre(x)
        x = self.conv_layers(x)
        x = self.head(x)
        return F.log_softmax(x, -1)