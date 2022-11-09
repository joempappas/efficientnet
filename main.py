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

def main():
    eff_0 = ef.EfficientNet(scale_width=1, scale_depth=1, output_size=1000)
    summary(eff_0, input_size=(64,3,224,224))
    print("Exiting")

if __name__ == "__main__":
    main()