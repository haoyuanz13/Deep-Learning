import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import utils as helper

'''
  Convolution Layers
  - contain two convolution layers and no pooling
'''
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(1, 1))
    self.conv2 = nn.Conv2d(16, 8, kernel_size=(7, 7), stride=(1, 1))

  def forward(self, x):
    # x = F.relu(self.conv1(x))
    x = helper.leaky_relu(self.conv1(x))
    x = self.conv2(x)
    # return F.relu(x)
    return helper.leaky_relu(x)

  def parameterSet(self, mu_w, std_w, bias):
    # initialize weights and bias
    weights = np.random.normal(mu_w, std_w, 2)
    self.conv1.weight.data.fill_(weights[0])
    self.conv2.weight.data.fill_(weights[1])

    self.conv1.bias.data.fill_(bias)
    self.conv2.bias.data.fill_(bias)

'''
  Fully Connected Layer for classification
  - contain one fully connected layer 
'''
class FCNetClassify(nn.Module):
  def __init__(self):
    super(FCNetClassify, self).__init__()
    # need to figure out the input size
    self.fc1 = nn.Linear(4 * 4 * 8, 1, bias=True)

  def forward(self, x):
    x = self.fc1(x)
    return F.sigmoid(x)

  def parameterSet(self, mu_w, std_w, bias):
    # initialize weights and bias
    weights = np.random.normal(mu_w, std_w, 1)
    self.fc1.weight.data.fill_(weights[0])
    self.fc1.bias.data.fill_(bias)

'''
  Fully Connected Layer for Region Regression
  - contain one fully connected layer
'''
class FCNetRegression(nn.Module):
  def __init__(self):
    super(FCNetRegression, self).__init__()
    self.fc1 = nn.Linear(4 * 4 * 8, 1, bias=True)

  def forward(self, x):
    x = self.fc1(x)
    return x

  def parameterSet(self, mu_w, std_w, bias):
    # initialize weights and bias
    weights = np.random.normal(mu_w, std_w, 1)
    self.fc1.weight.data.fill_(weights[0])
    self.fc1.bias.data.fill_(bias)

'''
  The combined network with common convolution layers and two branches
  - Classification branch
  - BB Regression branch
'''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv = ConvNet()
    self.fc_cls = FCNetClassify()
    self.fc_reg = FCNetRegression()

    self.loss_cls = None
    self.loss_reg = None

    self.cls_gt = None
    self.reg_gt = None

    self.loss_model_cls = nn.BCELoss()
    self.loss_model_reg = nn.MSELoss()

  def loss(self):
    return self.loss_cls + self.loss_reg

  def forward(self, x, cls_gt, reg_gt):
    self.cls_gt = cls_gt
    self.reg_gt = reg_gt

    x = self.conv(x)
    # reshape the image into feature vector
    x = x.view(-1, 4 * 4 * 8)
    # prediction for object class
    pred_cls = self.fc_cls(x)
    # prediction for region box
    pred_reg = self.fc_reg(x)

    self.loss_cls, self.loss_reg = self.loss_build(pred_cls, pred_reg)

    # return class and regression prediction respectively
    return pred_cls, pred_reg

  def parameterSet(self, mu_w, std_w, bias):
    # initialize weights and bias
    self.conv.parameterSet(mu_w, std_w, bias)
    self.fc_cls.parameterSet(mu_w, std_w, bias)
    self.fc_reg.parameterSet(mu_w, std_w, bias)

  # build loss
  def loss_build(self, pred_cls, pred_reg):
    # loss computation
    loss_cls = self.loss_model_cls(pred_cls, self.cls_gt)
    loss_reg = self.loss_model_reg(pred_reg, self.reg_gt)

    return loss_cls, loss_reg


# train process
def train_multi(data, cls_GT, reg_GT, model, iteration):
  opt = optim.SGD([ {'params': model.conv.parameters(), 'lr': 0.001},
                    {'params': model.fc_cls.parameters(), 'lr': 0.001},
                    {'params': model.fc_reg.parameters(), 'lr': 0.00001}])

  # num_instance = label_GT.data.shape[0]
  num_instance = cls_GT.data.shape[0]
  # loss_model_cls = nn.BCELoss()
  # loss_model_reg = nn.MSELoss()

  list_loss_cls, list_acc_cls = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([iteration, 1]), np.zeros([iteration, 1])

  # training loop
  i, threshold = 0, 0.5
  while (i < iteration):
    # clean up the gradient buffer
    opt.zero_grad()
    # forward to get prediction labels
    # cls_pred, reg_pred = model(data)
    cls_pred, reg_pred = model(data, cls_GT, reg_GT)
    # loss computation
    # loss_cls = loss_model_cls(cls_pred, cls_GT)
    # loss_reg = loss_model_reg(reg_pred, reg_GT)
    loss_comb = model.loss()

    # store loss value
    # list_loss_cls[i, 0] = loss_cls.data.numpy()[0]
    # list_loss_reg[i, 0] = loss_reg.data.numpy()[0]

    list_loss_cls[i, 0] = model.loss_cls.data.numpy()[0]
    list_loss_reg[i, 0] = model.loss_reg.data.numpy()[0]

    # backward to update parameters
    # loss_comb = loss_cls + 10 * loss_reg
    loss_comb.backward()
    # loss_cls.backward(retain_graph=True)
    # loss_reg.backward()
    opt.step()

    acc_cur_cls = helper.getAccuracy(cls_pred, cls_GT, num_instance, threshold)
    list_acc_cls[i, 0] = acc_cur_cls

    acc_cur_reg = helper.getAccuracy_reg(reg_pred, reg_GT, num_instance, threshold)
    list_acc_reg[i, 0] = acc_cur_reg

    print ('***************** The {}th iteration *******************'.format(i + 1))
    print ('--The cls loss {:.5f} and cls accuracy {:.5f}%.'.format(list_loss_cls[i, 0], acc_cur_cls * 100))
    print ('--The reg loss {:.5f} and reg accuracy {:.5f}%. \n'.format(list_loss_reg[i, 0], acc_cur_reg * 100))

    if (acc_cur_reg * 100 == 100.0):
      break    
    i += 1

  return list_loss_cls, list_acc_cls, list_loss_reg, list_acc_reg, i


def main():
  folder = "datasets/detection"
  train_npy = "detection_imgs.npy"
  cls_label_npy = "detection_labs.npy"
  reg_label_npy = "detection_width.npy"


  # the shape of training image is 64 x 16 x 16
  im_train = helper.dataloader(folder, train_npy)

  # convert into 4d matrix for images
  temp = np.zeros([64, 1, 16, 16])
  temp[:, 0, :, :] = im_train
  im_train = Variable(torch.from_numpy(temp).float())

  # corresponding class label (64, 1)
  cls_label = helper.dataloader(folder, cls_label_npy).reshape(-1, 64).transpose()
  cls_label = Variable(torch.from_numpy(cls_label).float())

  # corresponding region ground truth (64, 1)
  reg_label = helper.dataloader(folder, reg_label_npy).reshape(-1, 64).transpose()
  reg_label = Variable(torch.from_numpy(reg_label).float())
  
  model = Net()
  # model.parameterSet(mu_w=0, std_w=0.1, bias=0.1) 

  # train model
  iteration = 10000
  Loss_cls, Acc_cls, Loss_reg, Acc_reg, stop_ind = train_multi(im_train, cls_label, reg_label, model, iteration)
  
  title_cls, title_reg = "Loss & Accuracy - Iteration (Cls Branch)", "Loss & Accuracy - Iteration (Reg Branch)"
  helper.processPlot(stop_ind, Loss_cls[:stop_ind, :], Acc_cls[:stop_ind, :], title_cls)
  helper.processPlot(stop_ind, Loss_reg[:stop_ind, :], Acc_reg[:stop_ind, :], title_reg)


if __name__ == "__main__":
  main()
