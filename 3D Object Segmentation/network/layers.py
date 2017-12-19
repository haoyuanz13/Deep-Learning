'''
  File name: layers.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains Conv (e.g. MobileNet, ResNet) and Fc blocks
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pdb
  
shapeVisual = False

# max pooling layer
def maxPool(h, name):
  out = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], 
    strides=[1, 2, 2, 1], padding='SAME', name=name)

  if shapeVisual:
    print name, " | out shape: ", out.get_shape().as_list()
  
  return out


'''
  Weight decay 
'''
def varWeightDecay(var, wd_ration):
  weight_decay = tf.multiply(tf.nn.l2_loss(var), wd_ration, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)
  return var


'''
  The convolutional block: Conv + BN + Relu
'''
def ConvBlock(x, in_channels, out_channels, kernel_size, stride, is_train, reuse, wd=0.0, name='Conv'):  
  with tf.variable_scope(name, reuse=reuse) as scope:
    W = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels], 
      initializer = tf.truncated_normal_initializer(stddev=0.1))
    # weight decay if wd is larger than 0
    W = varWeightDecay(W, wd) if wd > 0 else W

    b = tf.get_variable('biases', [1, 1, 1, out_channels], 
      initializer = tf.constant_initializer(0.0))

    out = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b

    out = tf.contrib.layers.batch_norm(out, is_training=is_train, 
      scale=True, fused=True, updates_collections=None)

    out = tf.nn.relu(out)
  
  if shapeVisual:
    print name, " | out shape: ", out.get_shape().as_list()
  
  return out


'''
  The fully connected block: FC + BN + Relu
'''
def FcBlock(x, in_channels, out_channels, is_train, reuse, wd=0.0, name='FC'):
  with tf.variable_scope(name, reuse=reuse) as scope:
    W = tf.get_variable('weights', [in_channels, out_channels], 
      initializer = tf.truncated_normal_initializer(stddev=0.1))
    W = varWeightDecay(W, wd) if wd > 0 else W
      
    b = tf.get_variable('biases', [1, out_channels],
      initializer = tf.constant_initializer(0.0))

    out = tf.matmul(x, W)

    out = tf.contrib.layers.batch_norm(out, is_training=is_train, 
      scale=True, fused=True, updates_collections=None)
    
    out = tf.nn.relu(out)

  if shapeVisual:
    print name, " | out shape: ", out.get_shape().as_list()
  
  return out


'''
  Mobile net: contains depthwise and pointwise
'''
def MobileBloack(x, in_channels, out_channels, depth_kernel_size_h, depth_kernel_size_w, is_train, reuse, wd=0.0, name='Mobile'):
  with tf.variable_scope(name, reuse=reuse) as scope:
    '''
      depthwise conv
    '''
    W_depth = tf.get_variable('depth_weights', [depth_kernel_size_h, depth_kernel_size_w, in_channels, 1], 
      initializer = tf.truncated_normal_initializer(stddev=0.1))   
    # weight decay if wd is larger than 0
    W_depth = varWeightDecay(W_depth, wd) if wd > 0 else W_depth
    b_depth = tf.get_variable('depth_biases', [1, 1, 1, in_channels], initializer = tf.constant_initializer(0.0))

    # use tf.depthwise_conv2d
    out = tf.nn.depthwise_conv2d(x, W_depth, strides=[1, 1, 1, 1], padding='SAME') + b_depth

    out = tf.contrib.layers.batch_norm(out, is_training=is_train, scale=True, fused=True, updates_collections=None)
    out = tf.nn.relu(out)

    '''
      pointwise conv
    '''
    W_point = tf.get_variable('point_weights', [1, 1, in_channels, out_channels], 
      initializer = tf.truncated_normal_initializer(stddev=0.1))
    # weight decay if wd is larger than 0
    W_point = varWeightDecay(W_point, wd) if wd > 0 else W_point
    b_point = tf.get_variable('point_biases', [1, 1, 1, out_channels], initializer = tf.constant_initializer(0.0))

    out = tf.nn.conv2d(out, W_point, strides=[1, 1, 1, 1], padding='SAME') + b_point

    out = tf.contrib.layers.batch_norm(out, is_training=is_train, scale=True, fused=True, updates_collections=None)
    out = tf.nn.relu(out)

  if shapeVisual:
    print name, " | out shape: ", out.get_shape().as_list()
  
  return out


'''
  ResNet Block
'''
def ResBlock(x, in_channels, out_channels, weight1_size, weight2_size, identity_size, is_train, reuse, wd=0.0, name='Res'):
  with tf.variable_scope(name, reuse=reuse) as scope:
    '''
      weight layers 
    '''
    W_wl_1 = tf.get_variable('weight1', [weight1_size, weight1_size, in_channels, out_channels],
          initializer = tf.truncated_normal_initializer(stddev=0.1))
    W_wl_1 = varWeightDecay(W_wl_1, wd) if wd > 0 else W_wl_1
    b_wl_1 = tf.get_variable('bias1', [1, 1, 1, out_channels], initializer = tf.constant_initializer(0.0))

    x_w = tf.nn.conv2d(x, W_wl_1, strides=[1, 1, 1, 1], padding='SAME') + b_wl_1
    x_w = tf.contrib.layers.batch_norm(x_w, is_training=is_train, scale=True, fused=True, updates_collections=None)
    x_w = tf.nn.relu(x_w)

    W_wl_2 = tf.get_variable('weight2', [weight2_size, weight2_size, out_channels, out_channels],
          initializer = tf.truncated_normal_initializer(stddev=0.1))
    W_wl_2 = varWeightDecay(W_wl_2, wd) if wd > 0 else W_wl_2
    b_wl_2 = tf.get_variable('bias2', [1, 1, 1, out_channels], initializer = tf.constant_initializer(0.0))

    x_w = tf.nn.conv2d(x_w, W_wl_2, strides=[1, 1, 1, 1], padding='SAME') + b_wl_2
    x_w = tf.contrib.layers.batch_norm(x_w, is_training=is_train, scale=True, fused=True, updates_collections=None)

    '''
      identity layers
    '''
    W_idty = tf.get_variable('weight_idty', [identity_size, identity_size, in_channels, out_channels],
      initializer = tf.truncated_normal_initializer(stddev=0.1))
    W_idty = varWeightDecay(W_idty, wd) if wd > 0 else W_idty
    b_idty = tf.get_variable('bias_idty', [1, 1, 1, out_channels], initializer = tf.constant_initializer(0.0))

    x_idty = tf.nn.conv2d(x, W_idty, strides=[1, 1, 1, 1], padding='SAME') + b_idty
    x_idty = tf.contrib.layers.batch_norm(x_idty, is_training=is_train, scale=True, fused=True, updates_collections=None)

    # combine the identity and weighted outputs, then nonlinearize it 
    out = tf.nn.relu(x_idty + x_w)

  if shapeVisual:
    print name, " | out shape: ", out.get_shape().as_list()
  
  return out




