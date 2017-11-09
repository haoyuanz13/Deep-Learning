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
train_display_interval = 5
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
    mask_batch = torch.from_numpy(mask_batch).type(torch.FloatTensor)

    # get valid index (pos and neg)
    ind_pos, ind_neg = torch.eq(mask_batch, 1), torch.eq(mask_batch, 0)
    valid_mask = torch.ne(mask_batch, 2).type(torch.FloatTensor)
    validnum = float(ind_pos.sum() + ind_neg.sum())

    # convert to variable in cuda form
    im_batch, mask_batch = Variable(im_batch).cuda(1), Variable(mask_batch).cuda(1)
    valid_mask = Variable(valid_mask).cuda(1)

    # training process
    optimizer.zero_grad()
    feaMap = model(im_batch)

    # compute loss
    loss_feaMap, loss_mask = valid_mask * feaMap, valid_mask * mask_batch
    loss = loss_feaMap * loss_mask + (1 - loss_feaMap) * (1 - loss_mask)
    loss = -loss.log().sum() / validnum

    list_loss[batch_idx, 0] = loss.data[0]
    loss.backward()

    # count accuracy
    threshold_pos, threshold_neg = 0.75, 0.25
    predPos = torch.ge(feaMap, threshold_pos).type(torch.FloatTensor).cuda(1)
    predNeg = torch.ge(feaMap, threshold_neg).type(torch.FloatTensor).cuda(1)

    pos_match = predPos.eq(mask_batch).data * ind_pos.cuda(1)
    neg_match = predNeg.eq(mask_batch).data * ind_neg.cuda(1)

    correct = pos_match.sum() + neg_match.sum()
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

  # end all steps
  return list_loss, list_acc, grad_conv_front, grad_conv_back


'''
  test for the trained model
'''
def test(dir_model, test_imgs, test_masks):
  print "testing...................................."
  the_model = RPN()

  model_param = torch.load(dir_model)
  the_model.load_state_dict(model_param)

  the_model.cuda(1)
  the_model.eval()

  # test
  im_batch = torch.from_numpy(test_imgs).type(torch.FloatTensor)
  mask_batch = torch.from_numpy(test_masks).type(torch.FloatTensor)

  # get valid index (pos and neg)
  ind_pos, ind_neg = torch.eq(mask_batch, 1), torch.eq(mask_batch, 0)
  valid_mask = torch.ne(mask_batch, 2).type(torch.FloatTensor)
  validnum = float(ind_pos.sum() + ind_neg.sum())

  # convert to variable in cuda form
  im_batch, mask_batch = Variable(im_batch).cuda(1), Variable(mask_batch).cuda(1)
  valid_mask = Variable(valid_mask).cuda(1)

  # feed forward
  feaMap = the_model(im_batch)

  # compute loss
  loss_feaMap, loss_mask = valid_mask * feaMap, valid_mask * mask_batch
  loss = loss_feaMap * loss_mask + (1 - loss_feaMap) * (1 - loss_mask)
  test_loss = -loss.log().sum() / validnum

  # count accuracy
  threshold_pos, threshold_neg = 0.75, 0.25
  predPos = torch.ge(feaMap, threshold_pos).type(torch.FloatTensor).cuda(1)
  predNeg = torch.ge(feaMap, threshold_neg).type(torch.FloatTensor).cuda(1)

  pos_match = predPos.eq(mask_batch).data * ind_pos.cuda(1)
  neg_match = predNeg.eq(mask_batch).data * ind_neg.cuda(1)

  correct = pos_match.sum() + neg_match.sum()


  correct /= validnum

  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}% \n'.format(test_loss, 100. * correct))

  return test_loss, correct


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
    print('---------- The {}th train epoch has completed -------------'.format(epoch))

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
    if epoch % 1 == 0:
      # save model
      torch.save(model.state_dict(), model_path)

      # test trained model 
      test_loss, test_acc = test(model_path, test_imgs, test_masks)

      # store results     
      np.save(file_name_train, [train_loss[:epoch, :], train_acc[:epoch, :]])
      np.save(file_name_train_grad, [train_gradw_f[:epoch, :], train_gradw_b[:epoch, :]])
      np.save(file_name_test, [test_loss, test_acc])

    # update index
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





