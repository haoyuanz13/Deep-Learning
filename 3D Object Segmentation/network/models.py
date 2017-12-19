from __future__ import absolute_import
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
import tensorflow.contrib.slim as slim
from functools import partial
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


  # obtain all variables of the base net
  var_all = tf.contrib.framework.get_variables(base)

  return var_all, data_conv5


'''
  Fully Connected Layers for bbox regression bracnh
'''
def FullyConnectedNet(feaMap, input_channel, weight_decay_ratio=0.0, reuse=True, is_train=True, name='regFCN'):
  with tf.variable_scope(name, reuse=reuse) as fcn:
   # fully connected layer: convert from input_channel to 1024
    with tf.variable_scope('FC1', reuse=reuse) as fc1:
      feaMap = FcBlock(feaMap, input_channel, 1024, is_train=is_train, reuse=reuse, wd=weight_decay_ratio)

    # fully connected layer: convert from 1024 to 256
    with tf.variable_scope('FC2', reuse=reuse) as fc2:
      feaMap = FcBlock(feaMap, 1024, 256, is_train=is_train, reuse=reuse, wd=weight_decay_ratio)

    # fully connected layer: convert from 256 to 4
    with tf.variable_scope('FC3', reuse=reuse) as fc3:
      feaMap = FcBlock(feaMap, 256, 4, is_train=is_train, reuse=reuse, wd=weight_decay_ratio)

  # obtain all variables of fcn block
  var_all = tf.contrib.framework.get_variables(fcn)

  return var_all, feaMap



'''
  Deconvolutional layers for the depth branch
'''
def deConvBlocks(feaMap, reuse=True, is_train=True, name='depthDeconv'):
  # batch normalize layer
  batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
  bn = partial(batch_norm, is_training=is_train)

  # deconvolutional layer
  dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # stacked layer: dconv + bn + relu
  dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=tf.nn.relu, biases_initializer=None)

  # Deconv Blocks
  with tf.variable_scope(name, reuse=reuse) as deconvBlock:
    # 1st deconv: [N, 4, 4, 256] -> [N, 8, 8,128]
    feaMap = dconv_bn_relu(feaMap, 128, 3, 2)

    # 2nd deconv: [N, 8, 8, 128] -> [N, 16, 16,128]
    feaMap = dconv_bn_relu(feaMap, 128, 3, 2)

    # 3rd deconv: [N, 16, 16, 128] -> [N, 32, 32,64]
    feaMap = dconv_bn_relu(feaMap, 64, 3, 2)

    # 4th deconv: [N, 32, 32, 64] -> [N, 64, 64,32]
    feaMap = dconv_bn_relu(feaMap, 32, 3, 2)

    # 5th deconv: [N, 64, 64, 32] -> [N, 128, 128,1]
    feaMap = dconv_bn_relu(feaMap, 1, 3, 2)
  
  # ovtain variables
  var_all = tf.contrib.framework.get_variables(deconvBlock)

  return var_all, feaMap
