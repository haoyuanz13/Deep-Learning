import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pdb


'''
  The convolutional block: Conv + BN + Relu
'''
def ConvBlock(x, in_channels, out_channels, kernel_size, stride, is_train, reuse):  
  vs = tf.get_variable_scope()
  W = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, out_channels],
        initializer = tf.constant_initializer(0.0))

  x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b

  x = tf.contrib.layers.batch_norm(x, is_training=is_train, 
        scale=True, fused=True, scope=vs, updates_collections=None)

  x = tf.nn.relu(x)
  return x


'''
  The fully connected block: FC + BN + Relu
'''
def FcBlock(x, out_channels, is_train, reuse):
  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[1]

  W = tf.get_variable('weights', [in_channels, out_channels],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, out_channels],
        initializer = tf.constant_initializer(0.0))

  x = tf.matmul(x, W)
  x = batch_norm(x, is_train=is_train)
  x = tf.nn.relu(x)
  return x


# '''
#   Batch normalization
# '''
# def batch_norm(x, is_train=True, decay=0.99, epsilon=0.001):                                          
#   shape_x = x.get_shape().as_list()
  
#   beta = tf.get_variable('beta', shape_x[-1], initializer=tf.constant_initializer(0.0))
  
#   gamma = tf.get_variable('gamma', shape_x[-1], initializer=tf.constant_initializer(1.0))
#   moving_mean = tf.get_variable('moving_mean', shape_x[-1],
#                 initializer=tf.constant_initializer(0.0), trainable=False)
#   moving_var = tf.get_variable('moving_var', shape_x[-1],
#                initializer=tf.constant_initializer(1.0), trainable=False)


#   if is_train:
#     mean, var = tf.nn.moments(x, np.arange(len(shape_x) - 1), keep_dims=True)
#     mean = tf.reshape(mean, [mean.shape.as_list()[-1]])
#     var = tf.reshape(var, [var.shape.as_list()[-1]])

#     update_moving_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
#     update_moving_var = tf.assign(moving_var, moving_var * decay + shape_x[0] / (shape_x[0] - 1) * var * (1 - decay))
#     update_ops = [update_moving_mean, update_moving_var]

#     with tf.control_dependencies(update_ops):
#       return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

#   else:
#     mean = moving_mean
#     var = moving_var
#     return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)