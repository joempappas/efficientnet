#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:00:57 2022

@author: joe
"""

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, 
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU() if use_activation else nn.Identity()
        self.kernel_size = kernel_size
        
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
                                        nn.SiLU(),
                                        nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
                                        nn.Sigmoid())
        
    def forward(self, x):
        result = self.squeeze(x)
        result = self.excite(result)
        return result*x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob=drop_prob
        self.mode = "row"
        
    def forward(self, input):
        if (self.drop_prob == 0) or (not self.training):
            return input
        survival_rate = 1.0 - self.drop_prob
        if self.mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise
        
        """
        random = torch.rand(x.shape[0], 1, 1, 1, device=x.device) 
        mask = random > self.drop_prob
        x = x.div(1-self.drop_prob)
        x = x * mask
        return x
        """

class MBConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, drop_prob = 0, mb_type=6, r_factor=24):
        super(MBConvBlock, self).__init__()
        
        padding = kernel_size//2
        hidden_dims = mb_type*in_channels
        self.mb_type = mb_type
        self.use_skip_connect = (in_channels == out_channels) and (stride == 1)
        self.expand_block = ConvBlock(in_channels, hidden_dims, kernel_size=1, stride=1, padding=0) if mb_type==6 else None
        self.stochastic_depth = StochasticDepth(drop_prob=drop_prob)
        self.reduce_block = nn.Sequential(ConvBlock(hidden_dims, hidden_dims, kernel_size, stride=stride, padding=padding, groups=hidden_dims),
                                          SEBlock(hidden_dims, r_factor=r_factor),
                                          ConvBlock(hidden_dims, out_channels, kernel_size=1, stride=1, padding=0, use_activation=False)
                                          )
        
    def forward(self, x):
        if self.mb_type == 6: 
            x = self.expand_block(x)
        
        x = self.reduce_block(x)
        if (self.use_skip_connect):
            result = self.stochastic_depth(x)
            x = x + result
        return x
        
        