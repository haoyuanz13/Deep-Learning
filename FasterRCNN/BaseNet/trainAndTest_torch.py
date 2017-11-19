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
import time

from utils import * 

'''
  Different file contains various models
'''
from basemodel import *


# define global variables
train_display_interval = 10
batch_size = 100
step = 100
max_iteration = 500


'''
  training model
  - input dataload of training images and corresponding label
'''
def train(model, trainData, epoch, learning_rate):
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
  model.train()

  batch_idx = 0
  data_iter = iter(trainData)

  list_loss, list_acc = np.zeros([step, 1]), np.zeros([step, 1])
  grad_conv_front, grad_conv_back = np.zeros([step, 1]), np.zeros([step, 1])

  while batch_idx < step:
    # get training data
    data, label = data_iter.next()

    data = data.type(torch.FloatTensor)
    label = label.type(torch.LongTensor)

    # convert from Tensor to Variable
    # data, label_gt = Variable(data), Variable(label)
    data, label_gt = Variable(data).cuda(1), Variable(label).cuda(1)
    # training process
    optimizer.zero_grad()
    label_pred = model(data)  # the output is log softmax value

    # compute the accuracy
    pred = label_pred.data.max(1, keepdim = True)[1] # get the index of the max log-probability
    correct = pred.eq(label_gt.data.view_as(pred)).cpu().sum()
    acc = correct / float(batch_size)
    list_acc[batch_idx, 0] = acc
    
    # nll loss is suitable for n classification problem
    loss = F.nll_loss(label_pred, label_gt)
    loss.backward()
    
    # obtain gradients
    for name, param in model.named_parameters():
      if name == "model.0.0.weight":
        grad_conv_front[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

      elif name == "model.9.weight":
        grad_conv_back[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

    optimizer.step()

    list_loss[batch_idx, 0] = loss.data[0]

    batch_idx += 1

    if batch_idx % train_display_interval == 0:
      print('Train Epoch:{} [{}/{}]  Accuracy: {:.0f}%, Loss: {:.6f}'.format(
        epoch, batch_idx * len(data) , step * batch_size, 100. * acc, loss.data[0]))

  return list_loss, list_acc, grad_conv_front, grad_conv_back
  # return list_loss, list_acc


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
def main(processType):
  # load data
  train_data = dataloader(forTrain=True, processType=processType, bs=batch_size)
  test_data = dataloader(forTrain=False, processType=processType, bs=batch_size)

  # build model
  Net_ind = 3
  model = Net3()
  model.cuda(1)

  print '=========== The current Net{} using the data process type {} ==========='.format(Net_ind, processType)

  train_loss, train_acc = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  train_gradw_f, train_gradw_b = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  # training process
  epoch, learning_rate = 1, 0.001
  delta_lr = (0.001 - 0.0001) / 2000

  # model / files name
  model_path = './models/model{}_dataProcessType_{}.pth'.format(Net_ind, processType)
  file_name_train = './res/Net{}_train_res_processType_{}.npy'.format(Net_ind, processType)
  file_name_train_grad = './res/Net{}_train_res_grad_processType_{}.npy'.format(Net_ind, processType)
  file_name_test = './res/Net{}_test_res_processType_{}.npy'.format(Net_ind, processType)

  timeCost = 0  
  while epoch <= max_iteration:
    decayed_learning_rate = learning_rate * np.exp(- delta_lr * (epoch - 1))
    print('---------- The {}th train epoch is processing with learning rate {:.12f} --------------'.format(epoch, decayed_learning_rate))

    start_time = time.time()
    losses, accs, grad_w_f, grad_w_b = train(model, train_data, epoch, decayed_learning_rate)
    elapsed_time = time.time() - start_time
    print('---------- The {}th train epoch has completed with {} s time cost -------------'.format(epoch, elapsed_time))
    timeCost += elapsed_time

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

  print '\n======== The training process has completed! Total [{} epochs] using [time {} s] =========='.format(max_iteration, timeCost)


if __name__ == '__main__':

  '''
    processType represents different data pre-process type
    0: raw data
    1: normalize the raw data
    2: normalize + random flip raw data
    3: normalize + random flip + pad + crop raw data
  '''
  processType = 0
  
  '''
    train the model
  '''
  while processType < 1:
    main(processType)
    processType += 1

  '''
    plot train loss and acc curve as well as test result
  '''
  # plotCurve(2, processType)
  
  '''
    plot feature map in different ConvNet layers
  '''
  # featureMapVis(processType, batch_size)

