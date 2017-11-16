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
  Define layers
'''
# convolution layer
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max pooling layer
def maxPool(h):
  return tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
  The main RPN network
'''
def RPN(data, mask_batch, reg_batch, reuse):
  with tf.variable_scope('RPN', reuse=reuse) as rpn: 
    # Common shared ConvNet with variable space: BaseNet
    with tf.variable_scope('BaseNet', reuse=reuse) as base:
      # 1st convolution layer
      with tf.variable_scope('conv1', reuse=reuse):
        data = ConvBlock(data, 3, 32, 5, 1, is_train=True, reuse=reuse)
        data_conv1 = maxPool(data)

      # the 2nd convolution layer
      with tf.variable_scope('conv2', reuse=reuse):
        data_conv1 = ConvBlock(data_conv1, 32, 64, 5, 1, is_train=True, reuse=reuse)
        data_conv2 = maxPool(data_conv1)

      # the 3rd convolution layer
      with tf.variable_scope('conv3', reuse=reuse):
        data_conv2 = ConvBlock(data_conv2, 64, 128, 5, 1, is_train=True, reuse=reuse)
        data_conv3 = maxPool(data_conv2)

      # the 3rd convolution layer
      with tf.variable_scope('conv4', reuse=reuse):
        data_conv4 = ConvBlock(data_conv3, 128, 256, 3, 1, is_train=True, reuse=reuse)

      # intermediate layer
      with tf.variable_scope('intermediate', reuse=reuse):
        inter_map = ConvBlock(data_conv4, 256, 256, 3, 1, is_train=True, reuse=reuse)


    # Cls branch with variable space: cls_branch
    with tf.variable_scope('cls_branch', reuse=reuse) as cls_branch:
      # get cls feature map
      with tf.variable_scope('cls_map', reuse=reuse) as cls_map:
        Weight = tf.Variable(tf.truncated_normal([1, 1, 256, 1], stddev=0.1), name='W_cls')
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name='b_cls')
        cls_map = conv2d(inter_map, Weight) + bias
        cls_map = tf.reshape(cls_map, [-1, 6, 6])

      # compute cls loss (sigmoid cross entropy)
      with tf.variable_scope('cls_loss', reuse=reuse) as cls_branch_loss:
        # obtain valid places (pos and neg) in mask
        cond_cls = tf.not_equal(mask_batch, tf.constant(2, dtype=tf.float32))
        # compute the sigmoid cross entropy loss: choose loss where cond is 1 while select 0
        cross_entropy_classify = tf.reduce_sum(
          tf.where(cond_cls, tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_batch, logits=cls_map), tf.zeros_like(mask_batch)))
        # count the pos and neg numbers
        effect_area_cls = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cross_entropy_classify /= effect_area_cls

      # compute cls accuracy
      with tf.variable_scope('cls_acc', reuse=reuse) as cls_branch_acc:
        correct = tf.reduce_sum(tf.where(cond_cls, tf.cast(abs(mask_batch - tf.nn.sigmoid(cls_map)) < 0.5, tf.float32), tf.zeros_like(mask_batch)))
        effect_area = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cls_acc = correct / effect_area


    # Reg branch with variable space: reg_branch
    with tf.variable_scope('reg_branch', reuse=reuse) as reg_branch:
      # get reg feature map
      with tf.variable_scope('reg_map', reuse=reuse) as reg_map:
        Weight = tf.Variable(tf.truncated_normal([1, 1, 256, 3], stddev=0.1), name='W_reg')
        bias = tf.Variable(tf.constant([24., 24., 32.]), name='b_reg')
        reg_map = conv2d(inter_map, Weight) + bias

      # compute reg loss (smooth l1 loss)
      with tf.variable_scope('reg_loss', reuse=reuse) as reg_branch_loss:
        # normalize coordinates
        reg_map_norm = tf.stack([tf.divide(reg_map[:, :, :, 0], 32), tf.divide(reg_map[:, :, :, 1], 32), tf.log(reg_map[:, :, :, 2])])
        reg_map_norm = tf.transpose(reg_map_norm, [1, 2, 3, 0])
    
        reg_mask_norm = tf.stack([tf.divide(reg_batch[:, :, :, 0], 32), tf.divide(reg_batch[:, :, :, 1], 32), tf.log(reg_batch[:, :, :, 2])])
        reg_mask_norm = tf.transpose(reg_mask_norm, [1, 2, 3, 0])

        cond_reg = tf.not_equal(reg_batch, tf.constant(0, dtype=tf.float32))
        smooth_L1_loss_reg = tf.reduce_sum(
            tf.where(cond_reg, tf.losses.huber_loss(reg_map_norm, reg_mask_norm, reduction=tf.losses.Reduction.NONE), tf.zeros_like(reg_mask_norm)))

        effect_area_reg = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_mask_norm), tf.zeros_like(reg_mask_norm)))
        smooth_L1_loss_reg /= effect_area_reg

      # compute reg accuracy
      with tf.variable_scope('reg_acc', reuse=reuse) as reg_branch_acc:
        correct = tf.reduce_sum(tf.where(cond_reg, tf.cast(abs(reg_batch - reg_map) < 1, tf.float32), tf.zeros_like(reg_batch)))
        effect_area = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_batch), tf.zeros_like(reg_batch)))
        reg_acc = correct / effect_area
    
    # joint loss    
    comb_loss = smooth_L1_loss_reg + cross_entropy_classify


    # gather variables 
    var_base = tf.contrib.framework.get_variables(base)
    var_cls = tf.contrib.framework.get_variables(cls_branch)
    var_reg = tf.contrib.framework.get_variables(reg_branch)

  var_all = tf.contrib.framework.get_variables(rpn)
  return comb_loss, cross_entropy_classify, smooth_L1_loss_reg, cls_acc, reg_acc, var_base, var_cls, var_reg, var_all


def trainAndTest(dict_train, dict_test):
  image = tf.placeholder(tf.float32, [None, 48, 48, 3])  # x_image represents the input image
  mask_gt = tf.placeholder(tf.float32, [None, 6, 6])  # mask for classification
  reg_gt = tf.placeholder(tf.float32, [None, 6, 6, 3])  # the ground truth for proposal regression

  loss, cls_loss, reg_loss, cls_acc, reg_acc, var_base, var_cls, var_reg, var_all = RPN(image, mask_gt, reg_gt, None)

  epoch, batch_size, step, iteration = 0, 100, 100, 2000
  display_interval, test_interval = 10, 5

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.00001
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
  op = True
  main(op)