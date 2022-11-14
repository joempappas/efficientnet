#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:41:24 2022

@author: joe
"""

import math
import torch
from torch import nn
from torchinfo import summary
import Nets.EfficientNet as ef
import Nets.JoeNet as jn

def main():
    #eff_0 = ef.EfficientNet(scale_width=1, scale_depth=1, output_size=10)
    #path = 'trained_models/EffNet_11_13_epoch_3.pt'
    #eff_0.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    #eff_0 = ef.EfficientNet(scale_width=1, scale_depth=1, output_size=1000)
    #summary(eff_0, input_size=(64,3,224,224))
    #print(eff_0)
    
    
    joe = jn.JoeNet(scale_width=1, scale_depth=1, output_size=10)
    summary(joe, input_size=(64,3,224,224))
    
    """
    p = []
    print("Beginning of epoch 3")
    for name, param in eff_0.named_parameters():
        p.append(param)
        print(f'{name}: {param}')
    print("Ending of epoch 3")
    """
if __name__ == "__main__":
    main()