from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pdb

from functools import partial
from utils import *


# leaky relu with alfa=0.2
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
def conv2d(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, padding='SAME', stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv



# standard convolution layer with valid padding
def conv2d_valid(input_, output_dim, k_h=2, k_w=2, d_h=1, d_w=1, stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


# deconvolution
def deconv2d(input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
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
  instance normalization layer
'''
def instance_norm(input, name="instance_norm"):
  with tf.variable_scope(name):
    # obtain the channel number of input
    depth = input.get_shape()[3]

    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input - mean) * inv

    return scale * normalized + offset

'''
  Residule block
'''
def residule_block(x, dim, ks=3, s=1, name='resBlock'):
  p = int((ks - 1) / 2)
  y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
  y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c1'), name+'_bn1')
  y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
  y = instance_norm(conv2d(y, dim, ks, ks, s, s, padding='VALID', name=name+'_c2'), name+'_bn2')
  return y + x


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


'''
  L1 loss for in_ and target
'''
def abs_criterion(in_, target):
  return tf.reduce_mean(tf.abs(in_ - target))

'''
  Least square loss
'''
def mae_criterion(in_, target):
  return tf.reduce_mean((in_ - target)**2)

'''
  Log-likelihood loss
'''
def sce_criterion(logits, labels):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))