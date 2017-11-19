import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image

from layers import *
from helper import *
from RPN_section import *

'''
  load data as dictionary
'''
def dataProcess(readTest=False):
  dict_cur = {'img': [], 'cls_gt': [], 'reg_gt': []}
  
  # obtain data path (test or train)
  if readTest:
    print '>>>>>>>>>>> load training data .................'
    path = 'cifar10_transformed/devkit/test.txt'
  else:
    print '>>>>>>>>>>> load testing data .................'
    path = 'cifar10_transformed/devkit/train.txt'
  
  with open(path) as f:
    for line in f:
      file_info = line.split()
      file_name = file_info[0]

      image = 'cifar10_transformed/imgs/' + file_name
      mask_image = 'cifar10_transformed/masks/' + file_name

      dict_cur['img'].append(mpimg.imread(image))
      mask_cur = np.array(Image.open(mask_image))
      
      dict_cur['cls_gt'].append(mask_cur)

      x_y_w_map = np.zeros((6, 6, 3), dtype=np.float32)
      x_y_w_map[mask_cur == 1, 0] = file_info[2]
      x_y_w_map[mask_cur == 1, 1] = file_info[3]
      x_y_w_map[mask_cur == 1, 2] = file_info[4]
      
      dict_cur['reg_gt'].append(x_y_w_map)
  
  dict_cur['img'] = np.array(dict_cur['img'])
  dict_cur['cls_gt'] = np.array(dict_cur['cls_gt'])
  dict_cur['reg_gt'] = np.array(dict_cur['reg_gt'])

  return dict_cur


'''
  The train and test model
'''
def trainAndTest(dict_train, dict_test):
  image = tf.placeholder(tf.float32, [None, 48, 48, 3])  # x_image represents the input image
  mask_gt = tf.placeholder(tf.float32, [None, 6, 6])  # mask for classification
  reg_gt = tf.placeholder(tf.float32, [None, 6, 6, 3])  # the ground truth for proposal regression

  _, _, _, loss, cls_loss, reg_loss, cls_acc, reg_acc, var_base, var_cls, var_reg, var_all = RPN(image, mask_gt, reg_gt, None)

  epoch, batch_size, step, iteration = 0, 100, 100, 200
  display_interval, test_interval = 10, 5

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.00005
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, step, 1 - 10 ** (-iteration), staircase=True)

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_all)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_all, global_step=global_step)

  
  list_loss_cls, list_acc_cls = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([iteration, 1]), np.zeros([iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  while (epoch < iteration):
    print '\n********************************* The {}th epoch training is processing *********************************'.format(epoch + 1)
    # data shuffle 
    arr = np.arange(dict_train['img'].shape[0])
    np.random.shuffle(arr)

    # batch size data
    data_epoch = dict_train['img'][arr[:], :, :, :]
    mask_epoch = dict_train['cls_gt'][arr[:], :, :]
    reg_mask_epoch = dict_train['reg_gt'][arr[:], :, :, :]

    # train for each epoch
    cls_loss_sum, reg_loss_sum = 0, 0
    cls_acc_sum, reg_acc_sum = 0, 0

    step_cur = 0
    while (step_cur < step):
      # obtain current batch data for training
      data_batch = data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
      mask_batch = mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
      reg_mask_batch = reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]

      # evaluation
      train_loss_cls = cls_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch})
      train_acc_cls = cls_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch})

      train_loss_reg = reg_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch})
      train_acc_reg = reg_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch})

      cls_loss_sum += train_loss_cls
      reg_loss_sum += train_loss_reg

      cls_acc_sum += train_acc_cls
      reg_acc_sum += train_acc_reg

      if step_cur % display_interval == 0:
        print('Train Epoch:{} [{}/{}]  Cls Loss: {:.8f}, Accuracy: {:.2f}% || Reg Loss: {:.8f}, Accuracy: {:.4f}%'.format(
          epoch + 1, step_cur * batch_size , step * batch_size, train_loss_cls, 100. * train_acc_cls, train_loss_reg, 100. * train_acc_reg))

      # train the model
      train_step.run(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch})
      step_cur += 1

    # end of current epoch iteration 
    
    cls_loss_sum /= step
    reg_loss_sum /= step

    cls_acc_sum /= step
    reg_acc_sum /= step
    # print the result for each epoch
    print '\n********************************* The {}th epoch training has completed *********************************'.format(epoch + 1)
    print 'The Avg Cls Loss is {:.8f}, Avg Acc is {:.2f}% || The Avg Reg Loss is {:.8f}, Avg Acc is {:.2f}%. \n'.format(
      cls_loss_sum, 100 * cls_acc_sum, reg_loss_sum, 100 * reg_acc_sum)

    # store results
    list_loss_cls[epoch, 0], list_acc_cls[epoch, 0] = cls_loss_sum, cls_acc_sum
    list_loss_reg[epoch, 0], list_acc_reg[epoch, 0] = reg_loss_sum, reg_acc_sum


    '''
      test model
    '''
    test_cls_loss, test_cls_acc, test_reg_loss, test_reg_acc = 0, 0, 0, 0
    if (epoch + 1) % test_interval == 0:
      print '==========================================================================================================='
      print '--------------------------- [TEST] the trained model after {} epochs --------------------------------------'.format(epoch + 1)
      test_data_epoch = dict_test['img']
      test_mask_epoch = dict_test['cls_gt']
      test_reg_mask_epoch = dict_test['reg_gt']

      step_cur = 0
      while (step_cur < step):
        # obtain current batch data for training
        test_data_batch = test_data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
        test_mask_batch = test_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
        test_reg_mask_batch = test_reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]

        # evaluation
        test_cls_loss += cls_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch})
        test_cls_acc += cls_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch})

        test_reg_loss += reg_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch})
        test_reg_acc += reg_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch})

        step_cur += 1

      # end test
      test_cls_loss /= step
      test_cls_acc /= step
      test_reg_loss /= step
      test_reg_acc /= step

      print 'The Test Cls Loss is {:.8f}, Acc is {:.2f}% || The Test Reg Loss is {:.8f}, Acc is {:.2f}%.'.format(
        test_cls_loss, 100 * test_cls_acc, test_reg_loss, 100 * test_reg_acc)
      print '==========================================================================================================='

      # store results
      np.save('res/RPN_test_cls.npy', [test_cls_loss, test_cls_acc])
      np.save('res/RPN_test_reg.npy', [test_reg_loss, test_reg_acc])

      np.save('res/RPN_train_cls_loss.npy', list_loss_cls[:epoch + 1, :])
      np.save('res/RPN_train_cls_acc.npy', list_acc_cls[:epoch + 1, :])
      np.save('res/RPN_train_reg_loss.npy', list_loss_reg[:epoch + 1, :])
      np.save('res/RPN_train_reg_acc.npy', list_acc_reg[:epoch + 1, :])

    # update epoch
    epoch += 1

  # store results
  np.save('res/RPN_test_cls.npy', [test_cls_loss, test_cls_acc])
  np.save('res/RPN_test_reg.npy', [test_reg_loss, test_reg_acc])

  np.save('res/RPN_train_cls_loss.npy', list_loss_cls[:epoch, :])
  np.save('res/RPN_train_cls_acc.npy', list_acc_cls[:epoch, :])
  np.save('res/RPN_train_reg_loss.npy', list_loss_reg[:epoch, :])
  np.save('res/RPN_train_reg_acc.npy', list_acc_reg[:epoch, :])

  return


# main code
def main(showRes):
  if not showRes:
    # load data
    dict_train = dataProcess(readTest=False)
    dict_test = dataProcess(readTest=True)
    trainAndTest(dict_train, dict_test)

  else:
    processPlot()


if __name__ == "__main__":
  op = False
  main(op)