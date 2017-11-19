import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb

'''
  The network structure
  - 5 Conv layers, 5 max pooling layers and 1 fully connected layer
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''
class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    # Conv blocks: conv->BN->ReLU->max pooling
    def ConvBlock(c_in, c_out, ks, strs, pad):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=strs, padding=pad),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
      )

    # model for Net1
    self.model = nn.Sequential(
      ConvBlock(3, 32, (5, 5), (1, 1), 2),
      ConvBlock(32, 64, (5, 5), (1, 1), 2),
      ConvBlock(64, 128, (5, 5), (1, 1), 2),
      ConvBlock(128, 256, (5, 5), (1, 1), 2),
      ConvBlock(256, 512, (3, 3), (1, 1), 1),
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
  The MobileNet structure
  - 4 depthwise separable convolutional layers, 1 Conv layers
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''
class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    
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
  The ResNet structure
  - 4 ResNet convolutional layers, 1 Conv layers
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''
class ResBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super(ResBlock, self).__init__()
    self.identity_layer = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(1, 1), padding=0)
    self.weight_layer1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
    self.weight_layer2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=1)

    self.BN_identity = nn.BatchNorm2d(out_c)
    self.BN_weight1 = nn.BatchNorm2d(out_c)
    self.BN_weight2 = nn.BatchNorm2d(out_c)

    self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

  def forward(self, x):
    identity = self.BN_identity(self.identity_layer(x))
    
    x = F.relu(self.BN_weight1(self.weight_layer1(x)))
    x = self.BN_weight2(self.weight_layer2(x))
    
    x = F.relu(x + identity)
    x = self.maxPool(x)
    return x

class Net3(nn.Module):
  args = [(3, 32), (32, 64), (64, 128), (128, 256)]

  def __init__(self):
    super(Net3, self).__init__()
    self.res_Layers = self.layerBuild()

    # convolution layers: increase the channel but keep image size
    self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
    
    # batch normalization
    self.BN5 = nn.BatchNorm2d(512)

    # max pooling: downsample image by 2
    self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    # fully connected layers for softmax
    self.fc1 = nn.Linear(512, 10)

  # stack res blocks
  def layerBuild(self):
    layers = []

    for arg in self.args:
      inc, outc = arg[0], arg[1]
      layers.append(ResBlock(inc, outc))

    return nn.Sequential(*layers)


  def forward(self, x):
    x = self.res_Layers(x)

    # conv5 + bacth norm + relu + avg pool
    x = F.relu(self.BN5(self.conv5(x)))
    x = self.maxPool(x)

    # convert image into feature vector 
    x = x.view(-1, 512)

    # fully connected layer: 512d to 10d 
    x = self.fc1(x)

    # log soft max: prevent underflow
    return F.log_softmax(x)