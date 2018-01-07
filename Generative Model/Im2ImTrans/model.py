from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb

from functools import partial
from utils import *



# leaky relu function
# def lrelu(x, leak=0.2, name="lrelu"):
#   with tf.variable_scope(name):
#     f1 = 0.5 * (1 + leak)
#     f2 = 0.5 * (1 - leak)
#     return f1 * x + f2 * abs(x)


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak * x)


# fully connected layer
def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


# standard convolution layer
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv



# deconvolution
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):

  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv



'''
  batch normalization class
'''
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, 
      epsilon=self.epsilon, scale=True, scope=self.name)



'''
  fully connected layer
'''
def flatten_fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, normalizer_fn=None,
                            normalizer_params=None, weights_initializer=slim.xavier_initializer(), weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(), biases_regularizer=None, reuse=None, variables_collections=None,
                            outputs_collections=None, trainable=True, scope=None):
  with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
    if inputs.shape.ndims > 2:
      inputs = slim.flatten(inputs)
    
    return slim.fully_connected(inputs, num_outputs, activation_fn, normalizer_fn, normalizer_params, weights_initializer, weights_regularizer,
                  biases_initializer, biases_regularizer, reuse, variables_collections, outputs_collections, trainable, scope)

'''
  leaky relu
'''
def leak_relu(x, leak, scope=None):
  with tf.name_scope(scope, 'leak_relu', [x, leak]):
    y = tf.maximum(x, leak * x) if leak < 1 else tf.minimum(x, leak * x)
    return y



'''
  L2 reconstruction loss
'''
def l2_loss(predictions, real_values):
  """Return the loss operation between predictions and real_values.
  Add L2 weight decay term if any.
  Args:
      predictions: predicted values
      real_values: real values
  Returns:
      Loss tensor of type float.
  """
  with tf.variable_scope('loss'):
    # 1/2n \sum^{n}_{i=i}{(x_i - x'_i)^2}
    mse = tf.div(tf.reduce_mean(
        tf.square(tf.subtract(predictions, real_values))), 2, name="mse")
    tf.add_to_collection('losses', mse)
    
    # mse + weight_decay per layer
    error = tf.add_n(tf.get_collection('losses'), name='total_loss')

  return error
