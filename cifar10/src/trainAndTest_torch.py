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
# from p1_model_torch import *
from p2_model_torch import *


# define global variables
train_display_interval = 10
batch_size = 100
step = 100
max_iteration = 500


'''
  training model
  - input dataload of training images and corresponding label
'''
def train(model, trainData, epoch):
  optimizer = optim.SGD(model.parameters(), lr = 0.1)
  model.train()

  batch_idx = 0
  data_iter = iter(trainData)

  list_loss, list_acc = np.zeros([step, 1]), np.zeros([step, 1])
  grad_conv_front, grad_conv_back = np.zeros([step, 1]), np.zeros([step, 1])

  # load fake images
  img_path_1, im_gt_1 = './res_fig/p3/advImgs_1.npy', './res_fig/p3/advImgs_gt_1.npy'
  img_path_2, im_gt_2 = './res_fig/p3/advImgs_2.npy', './res_fig/p3/advImgs_gt_2.npy'

  img_1, label_1 = np.load(img_path_1), np.load(im_gt_1)
  img_2, label_2 = np.load(img_path_2), np.load(im_gt_2)

  while batch_idx < step:
    # select two random adversarial images
    # ind1, ind2 = random.randint(0, 9), random.randint(0, 9)
    ind1, ind2 = random.sample(range(0, 9), 5), random.sample(range(0, 9), 5)
    
    ind1 = np.asarray(ind1)
    ind2 = np.asarray(ind2)
    
    # create a fusion variable
    batch_fusion = np.zeros([110, 3, 32, 32])
    label_fusion = np.zeros([110])

    batch_fusion[0:5, :, :, :] = img_1[ind1, :, :, :]
    batch_fusion[5:10, :, :, :] = img_2[ind2, :, :, :]

    label_fusion[0:5] = label_1[ind1]
    label_fusion[5:10] = label_2[ind2]

    # get training data
    batch_cur, label_cur = data_iter.next()
    batch_fusion[10:, :, :, :] = batch_cur.numpy()
    label_fusion[10:] = label_cur.numpy()
    
    # convert to troch
    data = torch.from_numpy(batch_fusion)
    label = torch.from_numpy(label_fusion)

    data = data.type(torch.FloatTensor)
    label = label.type(torch.LongTensor)

    # convert from Tensor to Variable
    data, label_gt = Variable(data), Variable(label)
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
      if name == "conv1.weight":
        grad_conv_front[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

      elif name == "conv5.weight":
        grad_conv_back[batch_idx, 0] = torch.max(torch.abs(param.grad)).data[0]

    optimizer.step()

    list_loss[batch_idx, 0] = loss.data[0]

    batch_idx += 1

    if batch_idx % train_display_interval == 0:
      print('Train Epoch:{} [{}/{}]  Accuracy: {:.0f}%, Loss: {:.6f}'.format(
        epoch, batch_idx * len(data) , step * (batch_size + 10), 100. * acc, loss.data[0]))

  print('---------- The {}th train epoch has completed -------------'.format(epoch))
  return list_loss, list_acc, grad_conv_front, grad_conv_back
  # return list_loss, list_acc


'''
  test for the trained model
'''
def test(dir_model, testData):
  print "testing...................................."
  the_model = torch.load(dir_model)
  the_model.eval()

  test_loss, correct = 0, 0

  for data, label_gt in testData:
    data, label_gt = Variable(data), Variable(label_gt)
    label_pred = the_model(data)
    
    test_loss += F.nll_loss(label_pred, label_gt, size_average = False).data[0] # sum up batch loss
    
    pred = label_pred.data.max(1, keepdim = True)[1] # get the index of the max log-probability
    correct += pred.eq(label_gt.data.view_as(pred)).cpu().sum()
  
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
  # cifarImshow(train_data)

  # build model
  model = Net()

  train_loss, train_acc = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  train_gradw_f, train_gradw_b = np.zeros([max_iteration, 1]), np.zeros([max_iteration, 1])
  # training process
  epoch = 1
  while epoch <= max_iteration:
    print('---------- The {}th train epoch is processing --------------'.format(epoch))

    losses, accs, grad_w_f, grad_w_b = train(model, train_data, epoch)
    # losses, accs = train(model, train_data, epoch)

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

    if acc_avg >= 0.90 and abs(acc_avg - local_mean) < 0.005:
      break

    epoch += 1

    # save model and result every 100 epoch
    if epoch % 100 == 0:
      # save model
      model_path = './models/model_dataProcessType_{}_adv.pth'.format(processType)
      torch.save(model, model_path)

      # test trained model 
      test_loss, test_acc = test(model_path, test_data)

      file_name_train = 'train_res_processType_{}_adv.npy'.format(processType)
      np.save(file_name_train, [train_loss[:epoch, :], train_acc[:epoch, :]])

      file_name_train_grad = 'train_res_grad_processType_{}_adv.npy'.format(processType)
      np.save(file_name_train_grad, [train_gradw_f[:epoch, :], train_gradw_b[:epoch, :]])

      file_name_test = 'test_res_processType_{}_adv.npy'.format(processType)
      np.save(file_name_test, [test_loss, test_acc])



  # save model
  model_path = './models/model_dataProcessType_{}_adv.pth'.format(processType)
  torch.save(model, model_path)

  # test trained model 
  test_loss, test_acc = test(model_path, test_data)

  file_name_train = 'train_res_processType_{}_adv.npy'.format(processType)
  np.save(file_name_train, [train_loss[:epoch, :], train_acc[:epoch, :]])

  file_name_train_grad = 'train_res_grad_processType_{}_adv.npy'.format(processType)
  np.save(file_name_train_grad, [train_gradw_f[:epoch, :], train_gradw_b[:epoch, :]])

  file_name_test = 'test_res_processType_{}_adv.npy'.format(processType)
  np.save(file_name_test, [test_loss, test_acc])



if __name__ == '__main__':

  '''
    processType represents different data pre-process type
    0: raw data
    1: normalize the raw data
    2: normalize + random flip raw data
    3: normalize + random flip + pad + crop raw data
  '''
  processType = 1
  
  '''
    train the model
  '''
  while processType < 2:
    print 'The model for data process type {}'.format(processType)
    main(processType)
    processType += 1

  '''
    plot train loss and acc curve as well as test result
  '''
  plotCurve(processType)
  
  '''
    plot feature map in different ConvNet layers
  '''
  featureMapVis(processType, batch_size)


