'''
  Helper functions such as dataloader
'''

import numpy as np
import os
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# data loader (e.g. image, label)
def dataloader(folder, file_npy):
  path_file = os.path.join(folder, file_npy)
  data = np.load(path_file)

  return data

'''
  Onw designed L2 Loss
  - Input label_pred: prediction labels
  - Input label_GT: ground truth labels
  - Output loss: L2 loss with tensor type
'''
def L2Loss(label_pred, label_GT):
  total = label_GT.data.shape[0]

  lossl1 = torch.abs(torch.add(label_pred, -1 * label_GT))
  lossl2 = torch.pow(lossl1, 2)

  # sum up and normalize
  loss = torch.mean(lossl2)
  # loss = torch.sum(lossl2) / total
  return loss

def CrossEntropyLoss(label_pred, label_GT):
  total = label_GT.data.shape[0]

  temp1 = torch.mul(label_GT, torch.log(label_pred))
  temp2 = torch.mul(1 - label_GT, torch.log(1 - label_pred))

  loss = torch.add(temp1, temp2)
  # sum up and normalize
  loss = torch.mean(-1 * loss)
  
  return loss

'''
  Compute the accuracy of prediction
'''
def getAccuracy(label_pred, label_GT, num_instance, threshold):
  pred_numpy, gt_numpy = label_pred.data.numpy(), label_GT.data.numpy()

  pred_numpy = (pred_numpy > threshold)
  diff = pred_numpy - gt_numpy

  vote = diff[diff == 0].size

  acc = vote / float(num_instance)
  return acc

'''
  Compute the accuracy of regression prediction
'''
def getAccuracy_reg(reg_pred, reg_GT, num_instance, threshold):
  pred_numpy, gt_numpy = reg_pred.data.numpy(), reg_GT.data.numpy()

  # upper bound
  upper_gt = gt_numpy + threshold
  # lower bound
  lower_gt = gt_numpy - threshold

  ind_u, ind_l = (pred_numpy <= upper_gt), (pred_numpy >= lower_gt)

  ind_valid = ind_u * ind_l
  # return acc
  vote = pred_numpy[ind_valid].size

  acc = vote / float(num_instance)
  return acc

'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot(iteration, loss, accuracy, title):
  fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, loss)
  Ax0.set_title('Loss Value') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('loss')
  Ax0.grid(True)

  Ax1.plot(x, accuracy)
  Ax1.set_title('Prediction Accuracy')
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('Accuracy')
  Ax1.grid(True)

  plt.suptitle(title, fontsize=16)
  plt.show()


'''
  Customized Activation function: leakyReLU
'''
def leaky_relu(x):
  alpha = 0.25
  # positive part should have same result as relu(x)
  pos = F.relu(x)
  # negative part should have some negative values instead of 0
  abs_x = torch.abs(x)
  neg = alpha * (x - abs_x) * 0.5
  return pos + neg


'''
  Customized Activation function: ELU(Exponential Linear Unit)
'''
def ELU(x):
  alpha = 0.25
  # positive part should have same result as relu(x)
  pos = F.relu(x)
  # negative part should have some negative values instead of 0
  neg_x = (x - torch.abs(x)) * 0.5
  neg = alpha * (torch.exp(neg_x) - 1)
  return pos + neg


