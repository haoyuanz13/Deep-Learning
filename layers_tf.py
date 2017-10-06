'''
  Define convolution and fully connected layers (Tensorflow)
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

'''
  Convolution layer
'''
def ConvNet(x, out_channels, kernel_size, stride):
  # the size of x is [batch_size, im_height, im_width, channel]
  in_channels = x.get_shape()[3]

  W = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, int(in_channels), out_channels], stddev=0.1))
  b = tf.Variable(tf.constant(0.1, shape=[1, 1, 1, out_channels]))
  
  # convolution opeartion
  x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID') + b

  # nonlinear function to make model complex as well as functional
  x = tf.nn.relu(x)
  # the return size of x should have size [batch_size, new_im_height, new_im_width, out_channels]
  return x

'''
  Fully connected layer
'''
def FcNet(x, out_channels):
  # the size of x for fully connected layer is [batch_size, Height * Width * Channel_Number]
  in_channels = x.get_shape()[1]

  W = tf.Variable(tf.truncated_normal([int(in_channels), out_channels], stddev=0.1))
  b = tf.Variable(tf.constant(0.1, shape=[1, out_channels]))

  x = tf.matmul(x, W) + b

  # the output of x should have size [batch_size, out_channels] which is a 2D matrix
  return x


'''
  cross entropy
'''
def CrossEntropy(y_pred, y_gt):
  return -tf.reduce_mean(y_gt * tf.log(y_pred) + (1 - y_gt) * tf.log(1 - y_pred))

'''
  L2 loss
'''
def L2Loss(y_pred, y_gt):
  diff = tf.square(tf.subtract(y_pred, y_gt))
  loss = tf.reduce_mean(diff)

  return loss