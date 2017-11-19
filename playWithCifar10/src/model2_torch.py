import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable


'''
  The network structure (refer to AlexNet)
  - 5 Conv layers, 4 average pooling layers
  - Use ReLU as activation function
  - use softmax for classification
'''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # convolution layers
    self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.conv4 = nn.Conv2d(64, 32, kernel_size=(2, 2), stride=(1, 1), padding=0)
    self.conv5 = nn.Conv2d(32, 10, kernel_size=(2, 2), stride=(1, 1), padding=0)

    # average pooling: downsample image by 2
    self.avgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    # additional conv
    self.convAdd1 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd4 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd5 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd6 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd7 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd8 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd9 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd10 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd11 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd12 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd13 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd14 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd15 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd16 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd17 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd18 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd19 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
    self.convAdd20 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)


  def forward(self, x):
    # conv1 + relu + avg pool
    x = F.relu(self.conv1(x))
    x = self.avgPool(x)

    # additional conv nets
    identity1 = x
    x = F.relu(self.convAdd1(x))
    x += identity1

    identity2 = x
    x = F.relu(self.convAdd2(x))
    x += identity2

    identity3 = x
    x = F.relu(self.convAdd3(x))
    x += identity3

    identity4 = x
    x = F.relu(self.convAdd4(x))
    x += identity4

    identity5 = x
    x = F.relu(self.convAdd5(x))
    x += identity5

    identity6 = x
    x = F.relu(self.convAdd6(x))
    x += identity6

    identity7 = x
    x = F.relu(self.convAdd7(x))
    x += identity7

    identity8 = x
    x = F.relu(self.convAdd8(x))
    x += identity8

    identity9 = x
    x = F.relu(self.convAdd9(x))
    x += identity9

    identity10 = x
    x = F.relu(self.convAdd10(x))
    x += identity10 + identity1

    identity11 = x
    x = F.relu(self.convAdd11(x))
    x += identity11

    identity12 = x
    x = F.relu(self.convAdd12(x))
    x += identity12

    identity13 = x
    x = F.relu(self.convAdd13(x))
    x += identity13

    identity14 = x
    x = F.relu(self.convAdd14(x))
    x += identity14

    identity15 = x
    x = F.relu(self.convAdd15(x))
    x += identity15

    identity16 = x
    x = F.relu(self.convAdd16(x))
    x += identity16

    identity17 = x
    x = F.relu(self.convAdd17(x))
    x += identity17

    identity18 = x
    x = F.relu(self.convAdd18(x))
    x += identity18

    identity19 = x
    x = F.relu(self.convAdd19(x))
    x += identity19

    identity20 = x
    x = F.relu(self.convAdd20(x))
    x += identity20 + identity1
                                
    # conv2 + relu + avg pool
    # identity = x
    x = F.relu(self.conv2(x))
    # x += identity
    x = self.avgPool(x)

    # conv3 + relu + avg pool
    x = F.relu(self.conv3(x))
    x = self.avgPool(x)

    # conv4 + relu
    x = F.relu(self.conv4(x))

    # conv5 + relu + avg pool
    x = F.relu(self.conv5(x))
    x = self.avgPool(x)

    # convert image into feature vector 
    x = x.view(-1, 10 * 1 * 1)

    # log soft max: prevent underflow
    return F.log_softmax(x)