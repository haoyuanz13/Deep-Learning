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
def train(model, train_imgs, train_masks, train_reg, epoch, learning_rate):
  # build optimizer
  # optimizer = optim.Adam([ {'params': model.BaseNet.parameters(), 'lr': learning_rate},
  #                          {'params': model.prop_cls.parameters(), 'lr': learning_rate},
  #                          {'params': model.prop_reg.parameters(), 'lr': 100 * learning_rate}], weight_decay=0.05)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
  model.train()

  # array to store training results
  list_cls_loss, list_cls_acc = np.zeros([step, 1]), np.zeros([step, 1])
  list_reg_loss, list_reg_acc = np.zeros([step, 1]), np.zeros([step, 1])
  grad_conv_front, grad_conv_back = np.zeros([step, 1]), np.zeros([step, 1])

  batch_idx = 0
  while batch_idx < step:
    # otain the training data for one batch (images and masks)
    im_batch = train_imgs[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]
    mask_batch = train_masks[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]
    reg_batch = train_reg[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]

    # training process
    optimizer.zero_grad()

    # feed forward into the network
    cls_acc, reg_acc = model(im_batch, mask_batch, reg_batch)
    list_cls_acc[batch_idx, 0], list_reg_acc[batch_idx, 0] = cls_acc, reg_acc

    # obtain loss and back prop
    loss = model.combLoss()
    cls_loss, reg_loss = model.clsLoss().data[0], model.regLoss().data[0]
    list_cls_loss[batch_idx, 0], list_reg_loss[batch_idx, 0] = cls_loss, reg_loss

    loss.backward()

    # obtain gradients
    for name, param in model.named_parameters():
      if name == "BaseNet.0.0.weight":
        # pdb.set_trace()
        grad_conv_front[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

      elif name == "prop_reg.weight":
        grad_conv_back[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

    # update model parameters
    optimizer.step()

    # update idx
    batch_idx += 1

    if batch_idx % train_display_interval == 0:
      print('Train Epoch:{} [{}/{}]  Cls Accuracy: {:.0f}%, Loss: {:.6f} || Reg Diff Distance: {:.0f}, Loss: {:.6f}'.format(
        epoch, batch_idx * batch_size , step * batch_size, 100. * cls_acc, cls_loss, reg_acc, reg_loss))

  # end all steps
  return list_cls_loss, list_cls_acc, list_reg_loss, list_reg_acc, grad_conv_front, grad_conv_back


'''
  test for the trained model
'''
def test(dir_model, test_imgs, test_masks, test_reg):
  print "\n===========> testing...................................."
  the_model = RPN()

  model_param = torch.load(dir_model)
  the_model.load_state_dict(model_param)

  the_model.cuda(1)
  the_model.eval()

  # test dataset
  im_batch = test_imgs[:3000, :, :, :]
  mask_batch = test_masks[:3000, :, :, :]
  reg_batch = test_reg[:3000, :, :, :]

  # feed forward
  cls_acc, reg_acc = the_model(im_batch, mask_batch, reg_batch)
  cls_loss, reg_loss = the_model.clsLoss().data[0], the_model.regLoss().data[0]

  print('\nTest set: Average Cls loss: {:.4f}, Cls Accuracy: {:.4f}%'.format(cls_loss, 100. * cls_acc))
  print('Test set: Average Reg loss: {:.4f}, Reg Diff Distance: {:.4f} \n'.format(reg_loss, reg_acc))

  return cls_loss, cls_acc, reg_loss, reg_acc


'''
  main part: construct model, training and testing
'''
def main(train_imgs, train_masks, test_imgs, test_masks, train_reg, test_reg):
  print '************************* RPN Model Main Code *************************'
  # construct model
  model = RPN()
  model.parameterSet()
  model.cuda(1)

  # array to store training results including loss and accuracy
  train_cls_loss, train_cls_acc = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  train_reg_loss, train_reg_acc = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  train_gradw_f, train_gradw_b = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])

  # training process
  epoch, learning_rate = 1, 0.001
  delta_lr = (0.001 - 0.0001) / 2000

  # model / files name
  model_path = './models/model_RPN.pth'
  file_name_cls_train = './res/RPN_cls_train_res.npy'
  file_name_reg_train = './res/RPN_reg_train_res.npy'
  # file_name_train_grad = './res/RPN_cls_train_res_grad.npy'
  file_name_cls_test = './res/RPN_cls_test_res.npy'
  file_name_reg_test = './res/RPN_reg_test_res.npy'

  inds = np.arange(0, batch_size * step, 1)
  while epoch <= max_iteration:
    decayed_learning_rate = learning_rate * np.exp(- delta_lr * (epoch - 1))
    print('---------- The {}th train epoch is processing with learning rate {:.12f} --------------'.format(epoch, decayed_learning_rate))

    # shuffle data
    inds = np.random.permutation(inds)
    train_imgs_epoch, train_masks_epoch, train_reg_epoch = \
          train_imgs[inds, :, :, :], train_masks[inds, :, :, :], train_reg[inds, :, :, :]

    cls_losses, cls_accs, reg_losses, reg_accs, grad_w_f, grad_w_b = \
          train(model, train_imgs_epoch, train_masks_epoch, train_reg_epoch, epoch, decayed_learning_rate)

    print('--------------------------- The {}th train epoch has completed ------------------------------\n'.format(epoch))

    cls_loss_avg, cls_acc_avg = np.mean(cls_losses), np.mean(cls_accs)
    reg_loss_avg, reg_acc_avg = np.mean(reg_losses), np.mean(reg_accs)
    grad_w_f_avg, grad_w_b_avg = np.mean(grad_w_f), np.mean(grad_w_b)

    train_cls_loss[epoch - 1, :] = cls_loss_avg
    train_cls_acc[epoch - 1, :] = cls_acc_avg
    train_reg_loss[epoch - 1, :] = reg_loss_avg
    train_reg_acc[epoch - 1, :] = reg_acc_avg
    train_gradw_f[epoch - 1, :] = grad_w_f_avg
    train_gradw_b[epoch - 1, :] = grad_w_b_avg

    print('Avg Cls Accuracy: {:.4f}%, Avg Cls Loss: {:.4f}.'.format(100. * cls_acc_avg, cls_loss_avg))
    print('Avg Reg Accuracy: {:.4f}, Avg Reg Distance: {:.4f}.'.format(reg_acc_avg, reg_loss_avg))
    print('Avg grad wrt weight of the first Conv layer: {:.10f}.'.format(grad_w_f_avg))
    print('Avg grad wrt weight of the last Conv layer: {:.10f}.'.format(grad_w_b_avg))


    # if acc_avg >= 0.95:
    #   break
    
    # save model and result every 100 epoch
    if epoch % 1 == 0:
      # save model
      torch.save(model.state_dict(), model_path)

      # test trained model 
      test_cls_loss, test_cls_acc, test_reg_loss, test_reg_acc = \
          test(model_path, test_imgs, test_masks, test_reg)

      # store results     
      np.save(file_name_cls_train, [train_cls_loss[:epoch, :], train_cls_acc[:epoch, :]])
      np.save(file_name_reg_train, [train_reg_loss[:epoch, :], train_reg_acc[:epoch, :]])
      # np.save(file_name_train_grad, [train_gradw_f[:epoch, :], train_gradw_b[:epoch, :]])
      np.save(file_name_cls_test, [test_cls_loss, test_cls_acc])
      np.save(file_name_reg_test, [test_reg_loss, test_reg_acc])

    # update index
    epoch += 1



if __name__ == '__main__':
  # read in data
  print "================== RPN Training =================="

  print "----> loading training images and masks .............................."
  train_imgs = np.load("dataRPN/train_imgs.npy")
  train_masks = np.load("dataRPN/train_mask.npy")
  print "----> Completed! \n"

  print "----> loading test images and masks .................................."
  test_imgs = np.load("dataRPN/test_imgs.npy")
  test_masks = np.load("dataRPN/test_mask.npy")
  print "----> Completed! \n"

  print "----> loading training and test regression ground truth .............."
  train_reg = np.load("dataRPN/train_reg.npy")
  test_reg = np.load("dataRPN/test_reg.npy")
  print "----> Completed! \n"

  # the main code
  main(train_imgs, train_masks, test_imgs, test_masks, train_reg, test_reg)








