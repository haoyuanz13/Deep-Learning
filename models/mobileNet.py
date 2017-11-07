import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb

'''
  The MobileNet structure (Version 1)
  - 4 depthwise separable convolutional layers, 1 Conv layers
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''
class MobileNet1(nn.Module):
  def __init__(self):
    super(MobileNet, self).__init__()
    
    # depthwise separable convolution block (local size keeps same)
    def depthSepConv(c_in, c_out):
      return nn.Sequential(
        # depthwise conv
        nn.Conv2d(c_in, c_in, kernel_size=(5, 5), stride=(1, 1), padding=2, groups=c_in, bias=False),
        nn.BatchNorm2d(c_in),
        nn.ReLU(inplace=True),

        # pointwise conv
        nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
      )

    # max pooling (downsample by 2)
    self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    # mobilenet model
    self.model = nn.Sequential(
      depthSepConv(3, 32),
      self.maxPool,

      depthSepConv(32, 64),
      self.maxPool,

      depthSepConv(64, 128),
      self.maxPool,

      depthSepConv(128, 256),
      self.maxPool,

      # the fifth conv layer
      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      self.maxPool,
    )

    # fully connected layers for softmax
    self.fc1 = nn.Linear(512, 10)


  def forward(self, x):
    x = self.model(x)

    # convert image into feature vector 
    x = x.view(-1, 512)

    # fully connected layer: 512d to 10d 
    x = self.fc1(x)

    # log soft max: prevent underflow
    return F.log_softmax(x)




'''
  The MobileNet structure (Version 2)
  - 4 depthwise separable convolutional layers, 1 Conv layers
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''

'''
  The depthwise separable convolutional block
  The local size will be downsampled by 2 due to the max pooling layer
'''
class depthSepCon(nn.Module):
  def __init__(self, in_c, out_c, Dk):
    super(depthSepCon, self).__init__()
    self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=(Dk, Dk), stride=(1, 1), padding= (Dk - 1) / 2, groups=in_c, bias=False)
    self.BN_depth = nn.BatchNorm2d(in_c)

    self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
    self.BN_point = nn.BatchNorm2d(out_c)

    self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

  def forward(self, x):
    x = F.relu(self.BN_depth(self.depthwise(x)))
    x = F.relu(self.BN_point(self.pointwise(x)))
    x = self.maxPool(x)

    return x

class MobileNet2(nn.Module):
  # (a, b, c): a represents input channel, b is output channel, c is the kernel size in depthwise conv
  args = [(3, 32, 5), (32, 64, 5), (64, 128, 5), (128, 256, 5)]

  def __init__(self):
    super(Net2, self).__init__()
    # stacked depthwise separable layers
    self.mobile_layers = self.layerBuild()

    # the fifth conv layer
    self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)

    # batchnormalized layer for the fifth one
    self.BN5 = nn.BatchNorm2d(512)

    # max pooling: downsample image by 2
    self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    # fully connected layers for softmax
    self.fc1 = nn.Linear(512, 10)

  # stack depthwise separable conv layers together
  def layerBuild(self):
    layers = []

    for arg in self.args:
      inp, outp, dk = arg[0], arg[1], arg[2]
      layers.append(depthSepCon(inp, outp, dk))

    return nn.Sequential(*layers)


  def forward(self, x):
    x = self.mobile_layers(x)

    # conv5 + bacth norm + relu + avg pool
    x = F.relu(self.BN5(self.conv5(x)))
    x = self.maxPool(x)

    # convert image into feature vector 
    x = x.view(-1, 512)

    # fully connected layer: 512d to 10d 
    x = self.fc1(x)

    # log soft max: prevent underflow
    return F.log_softmax(x)