import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

'''
  The network structure
  - 3 Conv layers, 3 average pooling layers and 1 fully connected layer
  - Add batch normalization for each layer output
  - Use ReLU as activation function
  - use softmax for classification
'''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # convolution layers: increase the channel but keep image size
    self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2)
    
    # batch normalization
    self.batchNorm1 = nn.BatchNorm2d(32)
    self.batchNorm2 = nn.BatchNorm2d(32)
    self.batchNorm3 = nn.BatchNorm2d(64)
    self.batchNorm4 = nn.BatchNorm1d(64)

    # average pooling: downsample image by 2
    self.avgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    # fully connected layers
    self.fc1 = nn.Linear(64 * 4 * 4, 64)
    # 10d feature for softmax
    self.fc2 = nn.Linear(64, 10)


  def forward(self, x):
    # conv1 + bacth norm + relu + avg pool
    x = F.relu(self.batchNorm1(self.conv1(x)))
    x = self.avgPool(x)

    # conv2 + bacth norm + relu + avg pool
    x = F.relu(self.batchNorm2(self.conv2(x)))
    x = self.avgPool(x)

    # conv1 + bacth norm + relu + avg pool
    x = F.relu(self.batchNorm3(self.conv3(x)))
    x = self.avgPool(x)

    # convert image into feature vector 
    x = x.view(-1, 64 * 4 * 4)

    # fully connected layer: 64*4*4(d) to 64(d)
    x = F.relu(self.batchNorm4(self.fc1(x)))
    # fully connected layer: 64d to 10d 
    x = self.fc2(x)

    # log soft max: prevent underflow
    return F.log_softmax(x)
