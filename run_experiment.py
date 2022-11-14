# -*- coding: utf-8 -*-

import torch
import Nets.EfficientNet as ef
import math
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import InterpolationMode

eff_0 = ef.EfficientNet(scale_width=1, scale_depth=1, output_size=10)
path = 'trained_models/EffNet_11_10_epoch_10.pt'
eff_0.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def test_cifar(classifier, epoch):

  classifier.eval() 

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for images, targets in test_loader:
      images = images
      targets = targets
      output = classifier(images)
      test_loss += F.cross_entropy(output, targets, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1] 
      correct += pred.eq(targets.data.view_as(pred)).sum() 
  
  test_loss /= len(test_loader.dataset)

  print(f'Test result on epoch {epoch}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(test_loader.dataset)}%')




transform = torchvision.transforms.Compose([
    torchvision.transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10, 
                                       interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(224)])

#train_dataset = torchvision.datasets.CIFAR10('/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
# ----------- <End Your code> ---------------

# Define train_loader and test_loader
# ----------- <Your code> ---------------
batch_size = 64

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

batch_idx, (images, targets) = next(enumerate(test_loader))

output = eff_0(images)
test_loss = F.cross_entropy(output, targets, reduction='sum').item()
pred = output.data.max(1, keepdim=True)[1]
x = targets.data.view_as(pred)
correct = pred.eq(x).sum()
print(x)
print(pred)
print(correct)
print(targets)