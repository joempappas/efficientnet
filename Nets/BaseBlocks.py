#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:00:57 2022

@author: joe
"""

import torch
from torch import nn

class ConvBlock(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, 
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU() if use_activation else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x
    
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, r_factor=24):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        hidden_dim = in_channels // r_factor
        self.excite = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                        nn.SiLu(),
                                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                        nn.SiLU())
        
        def forward(self, x):
            resid = x
            x = self.squeeze(x)
            x = self.excite(x)
            return resid*x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob=drop_prob
        
    def forward(self, x):
        if (self.drop_prob == 0) or (not self.training):
            return x
        
        random = torch.rand(x.shape[0], 1, 1, 1, device=x.device) 
        mask = random > self.drop_prob
        x = x.div(1-self.drop_prob)
        x = x * mask
        return x

class MBConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, drop_prob, mb_type=6):
        super(MBConvBlock, self).__init__()
        
        padding = kernel_size//2
        hidden_dims = mb_type*in_channels
        self.use_skip_connect = (in_channels == out_channels) and (stride == 1)
        self.expand_block = ConvBlock(in_channels, hidden_dims, kernel_size, stride, padding) if mb_type==6 else None
        self.stochastic_depth = StochasticDepth(drop_prob=drop_prob)
        self.reduce_block = nn.Sequential(ConvBlock(hidden_dims, hidden_dims, kernel_size, stride, padding, groups=hidden_dims),
                                          SEBlock(in_channels, r_factor=24),
                                          ConvBlock(hidden_dims, out_channels, kernel_size, stride, padding, use_activation=False)
                                          )
        
    def forward(self, x):
        resid = x
        if self.mb_type == 6: self.expand_block(x)
        x = self.reduce_block(x)
        if (self.use_skip_connect):
            x = self.stochastic_depth(x)
            x = x + resid
        return x
        
        