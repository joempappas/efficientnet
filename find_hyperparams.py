# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
import math
import numpy as np
import torch
from torch import nn
import Nets.JoeNet as jn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

from torchvision.transforms.functional import InterpolationMode
transform = torchvision.transforms.Compose([
    torchvision.transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10, 
                                       interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(224)])

train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
# ----------- <End Your code> ---------------

# Define train_loader and test_loader
# ----------- <Your code> ---------------
batch_size = 200

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

batch_idx, (images, targets) = next(enumerate(train_loader))

grid_params = {
    'scale_depth': np.linspace(1.0, 2.0, int(1.0/0.05)),
    'scale_width': np.linspace(1.0, 1.5, int(0.5/0.05))
}

joenet_GS = GridSearchCV(jn.JoeNet(),
                      grid_params,
                      scoring=F.cross_entropy,
                      cv = 5)

result = joenet_GS.fit(images, targets)


####################         END CODE          ####################

print(f'The best parameters are {joenet_GS.best_params_}')
print(f'The best accuracy on the training data is {joenet_GS.score(images, targets)}')