import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb


'''
  The Region Proposal Network(RPN)
  - includes the base conv layers to extract feature map and 
  - additional proposal layers for classification and region regression
'''
class RPN(nn.Module):
  def __init__(self):
    super(RPN, self).__init__()
    # Conv blocks: conv->BN->ReLU->max pooling
    def ConvBlockWithMaxPool(c_in, c_out, ks, strs, pad):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=strs, padding=pad),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
      )

    # Conv blocks: conv->BN->ReLU
    def ConvBlockNoMaxPool(c_in, c_out, ks, strs, pad):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=strs, padding=pad),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
      )

    # model for Net1
    self.BaseNet = nn.Sequential(
      ConvBlockWithMaxPool(3, 32, (5, 5), (1, 1), 2),
      ConvBlockWithMaxPool(32, 64, (5, 5), (1, 1), 2),
      ConvBlockWithMaxPool(64, 128, (5, 5), (1, 1), 2),
      ConvBlockNoMaxPool(128, 256, (3, 3), (1, 1), 1)
    )

    # proposal classification branch
    self.prop_cls = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    # proposal regression branch
    self.prop_reg = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)


  def forward(self, x):
    x = self.BaseNet(x)

    # feature map for region proposal classification
    cls_map = self.prop_cls(x)
    
    return F.sigmoid(cls_map)

  # initialize bias
  def parameterSet(self):
    # initialize weights and bias
    self.prop_reg.bias.data[0] = 24
    self.prop_reg.bias.data[1] = 24
    self.prop_reg.bias.data[2] = 32


