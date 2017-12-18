'''
  File name: models.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains models (eg. BaseNet, RPN, fasterRCNN)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pdb

from layers import *


'''
  The BaseNet for feature extraction
  - Input view: one of the captured views from 3d object
  - Input input_channel: the number of view channel (eg. 1 (grayscale) or 3(rgb))
  - Input weight_decay_ratio: the weight decay ration, set as 0 if no need to set weight decay
  - Input ind_base: represents the conv block type (0:ConvNet, 1:MobileNet, 2:ResNet)
'''
def BaseNet(view, input_channel, weight_decay_ratio=0.0, ind_base=0, reuse=True, is_train=True, name='Base'):
  with tf.variable_scope(name, reuse=reuse) as base:
    # 1st convolution layer
    with tf.variable_scope('conv1', reuse=reuse):
      if ind_base == 0:
        view = ConvBlock(view, input_channel, 32, 5, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ConvNet1')
      elif ind_base == 1:
        view = MobileBloack(view, input_channel, 32, 5, 5, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='MobileNet1')
      else:
        view = ResBlock(view, input_channel, 32, 3, 3, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ResNet1')
      
      data_conv1 = maxPool(view, name='maxPooling1')

    # the 2nd convolution layer
    with tf.variable_scope('conv2', reuse=reuse):
      if ind_base == 0:
        data_conv1 = ConvBlock(data_conv1, 32, 64, 5, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ConvNet2')
      elif ind_base == 1:
        data_conv1 = MobileBloack(data_conv1, 32, 64, 5, 5, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='MobileNet2')
      else:
        data_conv1 = ResBlock(data_conv1, 32, 64, 3, 3, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ResNet2')
      
      data_conv2 = maxPool(data_conv1, name='maxPooling2')

    # the 3rd convolution layer
    with tf.variable_scope('conv3', reuse=reuse):
      if ind_base == 0:
        data_conv2 = ConvBlock(data_conv2, 64, 128, 5, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ConvNet3')
      elif ind_base == 1:
        data_conv2 = MobileBloack(data_conv2, 64, 128, 5, 5, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='MobileNet3')
      else:
        data_conv2 = ResBlock(data_conv2, 64, 128, 3, 3, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ResNet3')
      
      data_conv3 = maxPool(data_conv2, name='maxPooling3')

    # the 4th convolution layer
    with tf.variable_scope('conv4', reuse=reuse):
      if ind_base == 0:
        data_conv3 = ConvBlock(data_conv3, 128, 128, 5, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ConvNet4')
      elif ind_base == 1:
        data_conv3 = MobileBloack(data_conv3, 128, 128, 5, 5, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='MobileNet4')
      else:
        data_conv3 = ResBlock(data_conv3, 128, 128, 3, 3, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ResNet4')
      
      data_conv4 = maxPool(data_conv3, name='maxPooling4')

    # the 5th convolution layer
    with tf.variable_scope('conv5', reuse=reuse):
      if ind_base == 0:
        data_conv4 = ConvBlock(data_conv4, 128, 256, 5, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ConvNet5')
      elif ind_base == 1:
        data_conv4 = MobileBloack(data_conv4, 128, 256, 5, 5, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='MobileNet5')
      else:
        data_conv4 = ResBlock(data_conv4, 128, 256, 3, 3, 1, is_train=is_train, reuse=reuse, wd=weight_decay_ratio, name='ResNet5')
      
      data_conv5 = maxPool(data_conv4, name='maxPooling5')


  # obtain all variables of faster rcnn
  var_all = tf.contrib.framework.get_variables(base)

  return var_all, data_conv5