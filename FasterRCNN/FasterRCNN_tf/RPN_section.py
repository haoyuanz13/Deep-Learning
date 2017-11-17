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

'''
  The RPN network
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

  # return the feature map from conv4, cls_map, reg_map and all loss, accuracy and variables
  return data_conv4, cls_map, reg_map, comb_loss, cross_entropy_classify, smooth_L1_loss_reg, cls_acc, reg_acc, var_base, var_cls, var_reg, var_all