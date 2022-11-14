# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
from torchinfo import summary
import Nets.EfficientNet as ef
import Nets.JoeNet as jn


def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  alpha=1.6
  beta=1.1
  gamma=1.02
  
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'joenet-b0': (1.0, 1.0, 224),
      'joenet-b1': (1.0, 1.1, 240),
      'joenet-b2': (1.1, 1.2, 260),
      'joenet-b3': (1.2, 1.4, 300),
      'joenet-b4': (1.4, 1.8, 380),
      'joenet-b5': (1.6, 2.2, 456),
      'joenet-b6': (1.8, 2.6, 528),
      'joenet-b7': (2.0, 3.1, 600),
  }
  return params_dict[model_name]

def main():
    
    
    
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