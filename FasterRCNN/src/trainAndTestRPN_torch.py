import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb
import numpy as np
import matplotlib.pyplot as plt
import random

from utils import * 

'''
  Different file contains various models
'''
from p2_model import *


# define global variables
train_display_interval = 10
batch_size = 100
step = 100
max_iteration = 1000


'''
  training model
  - input dataload of training images and corresponding label
'''
def train(model, train_imgs, train_masks, epoch, learning_rate):
  # build optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
  model.train()

  # array to store training results
  list_loss, list_acc = np.zeros([step, 1]), np.zeros([step, 1])
  grad_conv_front, grad_conv_back = np.zeros([step, 1]), np.zeros([step, 1])

  batch_idx = 0
  while batch_idx < step:
    # otain the training data for one batch (images and masks)
    im_batch = train_imgs[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]
    mask_batch = train_masks[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]

    im_batch = torch.from_numpy(im_batch).type(torch.FloatTensor)
    mask_batch = torch.from_numpy(mask_batch).type(torch.LongTensor)

    # get valid index (pos and neg)
    ind_pos, ind_neg = torch.eq(mask_batch, 1), torch.eq(mask_batch, 0)
    validnum = float(ind_pos.sum() + ind_neg.sum())

    # convert to variable in cuda form
    im_batch, mask_batch = Variable(im_batch).cuda(1), Variable(mask_batch).cuda(1)

    # build one more channel for binary classification
    feaMap_2c = torch.FloatTensor(batch_size, 2, 6, 6).zero_()

    # training process
    optimizer.zero_grad()
    feaMap = model(im_batch)

    feaMap_2c[:, 0, :, :] = 1 - feaMap.data  # represents the neg class
    feaMap_2c[:, 1, :, :] = feaMap.data  # represents the pos class

    feaMap_2c = Variable(feaMap_2c, requires_grad=True).cuda(1).log()

    # nll loss is suitable for n classification problem
    loss = F.nll_loss(feaMap_2c, mask_batch[:, 0, :, :], ignore_index=2)
    list_loss[batch_idx, 0] = loss.data[0]
    loss.backward()

    # compute the accuracy
    pred = feaMap_2c.data.max(1, keepdim = True)[1] # get the index of the max log-probability
    correct = pred.eq(mask_batch.data.view_as(pred)).sum()
    acc = correct / validnum
    list_acc[batch_idx, 0] = acc

    # obtain gradients
    for name, param in model.named_parameters():
      if name == "BaseNet.0.0.weight":
        grad_conv_front[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

      elif name == "prop_cls.weight":
        grad_conv_back[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

    # update model parameters
    optimizer.step()

    # update idx
    batch_idx += 1

    if batch_idx % train_display_interval == 0:
      print('Train Epoch:{} [{}/{}]  Accuracy: {:.0f}%, Loss: {:.6f}'.format(
        epoch, batch_idx * batch_size , step * batch_size, 100. * acc, loss.data[0]))

  print('---------- The {}th train epoch has completed -------------'.format(epoch))
  return list_loss, list_acc, grad_conv_front, grad_conv_back
  

'''
  test for the trained model
'''
def test(dir_model, testData):
  print "testing...................................."
  the_model = Net3()

  model_param = torch.load(dir_model)
  the_model.load_state_dict(model_param)

  the_model.cuda(1)
  the_model.eval()

  test_loss, correct = 0, 0

  for data, label_gt in testData:
    # data, label_gt = Variable(data), Variable(label_gt)
    data, label_gt = Variable(data).cuda(1), Variable(label_gt).cuda(1)
    label_pred = the_model(data)
    
    test_loss += F.nll_loss(label_pred, label_gt, size_average = False).data[0] # sum up batch loss
    
    pred = label_pred.data.max(1, keepdim = True)[1] # get the index of the max log-probability
    # correct += pred.eq(label_gt.data.view_as(pred)).cpu().sum()
    correct += pred.eq(label_gt.data.view_as(pred)).cuda(1).sum()
  
  total = float(len(testData.dataset))
  test_loss /= total

  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}% \n'.format(test_loss, 100. * correct / total))

  return test_loss, correct / total


'''
  main part: construct model, training and testing
'''
def main(train_imgs, train_masks, test_imgs, test_masks):
  print '************************* RPN Model Main Code *************************'
  # construct model
  model = RPN()
  model.cuda(1)

  # array to store training results including loss and accuracy
  train_loss, train_acc = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  train_gradw_f, train_gradw_b = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])

  # training process
  epoch, learning_rate = 1, 0.001
  delta_lr = (0.001 - 0.0001) / 2000

  # model / files name
  model_path = './models/model_RPN_cls.pth'
  file_name_train = './res/RPN_cls_train_res.npy'
  file_name_train_grad = './res/RPN_cls_train_res_grad.npy'
  file_name_test = './res/RPN_cls_test_res.npy'

  inds = np.arange(0, batch_size * step, 1)
  while epoch <= max_iteration:
    decayed_learning_rate = learning_rate * np.exp(- delta_lr * (epoch - 1))
    print('---------- The {}th train epoch is processing with learning rate {:.12f} --------------'.format(epoch, decayed_learning_rate))

    # shuffle data
    inds = np.random.permutation(inds)
    train_imgs, train_masks = train_imgs[inds, :, :, :], train_masks[inds, :, :, :]
    losses, accs, grad_w_f, grad_w_b = train(model, train_imgs, train_masks, epoch, decayed_learning_rate)

    loss_avg, acc_avg = np.mean(losses), np.mean(accs)
    grad_w_f_avg, grad_w_b_avg = np.mean(grad_w_f), np.mean(grad_w_b)

    train_loss[epoch - 1, :] = loss_avg
    train_acc[epoch - 1, :] = acc_avg
    train_gradw_f[epoch - 1, :] = grad_w_f_avg
    train_gradw_b[epoch - 1, :] = grad_w_b_avg

    print('Avg Accuracy: {:.4f}%, Avg Loss: {:.4f}.'.format(100. * acc_avg, loss_avg))
    print('Avg grad wrt weight of the first Conv layer: {:.10f}.'.format(grad_w_f_avg))
    print('Avg grad wrt weight of the last Conv layer: {:.10f}.'.format(grad_w_b_avg))

    # test whether jump into local optimal
    local_mean = np.mean(train_acc[epoch - 20 : epoch - 1]) if epoch >= 20 else 0
    print 'The current local mean acc is {:.4f}% \n'.format(100 * local_mean)

    if acc_avg >= 0.95 and abs(acc_avg - local_mean) < 0.001:
      break
    
    # save model and result every 100 epoch
    if epoch % 100 == 0:
      # save model
      torch.save(model.state_dict(), model_path)

      # test trained model 
      test_loss, test_acc = test(model_path, test_data)  
      # test_loss, test_acc = testnow(model, test_data)    
      np.save(file_name_train, [train_loss[:epoch, :], train_acc[:epoch, :]])
      np.save(file_name_train_grad, [train_gradw_f[:epoch, :], train_gradw_b[:epoch, :]])
      np.save(file_name_test, [test_loss, test_acc])


    epoch += 1



if __name__ == '__main__':
  # read in data
  print "================== RPN Training =================="

  print "----> loading training images and masks .........."
  train_imgs = np.load("dataRPN/train_imgs.npy")
  train_masks = np.load("dataRPN/train_mask.npy")
  print "----> Completed! \n"

  print "----> loading test images and masks .............."
  test_imgs = np.load("dataRPN/test_imgs.npy")
  test_masks = np.load("dataRPN/test_mask.npy")
  print "----> Completed! \n"

  # the main code
  main(train_imgs, train_masks, test_imgs, test_masks)


