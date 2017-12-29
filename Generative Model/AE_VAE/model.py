from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial
from utils import *

'''
  Generative Model G
  - Input z: random noise vector
  - Output net: generated map with same size as data
'''
def generator(z, dim=64, reuse=True, training=True):
  # batch normalize layer
  batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
  bn = partial(batch_norm, is_training=training)

  # deconvolutional layer
  dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # stacked layer: dconv + bn + relu
  dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=tf.nn.relu, biases_initializer=None)

  # fully connected layer
  fc = partial(flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # stacked layer with fc 
  fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=tf.nn.relu, biases_initializer=None)

  # model G
  with tf.variable_scope('Model_G', reuse=reuse):
    # enlarge z dimension and reshape into a matrix
    y = fc_bn_relu(z, 4 * 4 * dim * 8)
    y = tf.reshape(y, [-1, 4, 4, dim * 8])

    # 1st deconv: dim*8 -> dim*4
    y = dconv_bn_relu(y, dim * 4, 5, 2)

    # 2nd deconv: dim*4 -> dim*2
    y = dconv_bn_relu(y, dim * 2, 5, 2)

    # 3rd deconv: dim*2 -> dim
    y = dconv_bn_relu(y, dim * 1, 5, 2)

    # 4th deconv: dim -> 1
    img = tf.tanh(dconv(y, 1, 5, 2))
    return img


'''
  Discriminator model D (DCGAN)
  - Input img: image data from dataset or the G model
  - Output logit: the scalar to represent the prob that net belongs to the real data
'''
def discriminator_bn(img, dim=64, reuse=True, training=True):
  # batch normalize layer
  batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
  bn = partial(batch_norm, is_training=training)

  # standard convolutional layer using lrelu  
  conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
  lrelu = partial(leak_relu, leak=0.2)

  # stacked layer: conv + bn + leaky relu
  conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

  # fully connected layer
  fc = partial(flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # model D
  with tf.variable_scope('Model_D', reuse=reuse):
    # 1st conv: 1 -> dim
    y = lrelu(conv(img, dim, 5, 2))

    # 2nd conv: dim -> dim*2
    y = conv_bn_lrelu(y, dim * 2, 5, 2)

    # 3rd conv: dim*2 -> dim*4
    y = conv_bn_lrelu(y, dim * 4, 5, 2)

    # 4th conv: dim*4 -> dim*8
    y = conv_bn_lrelu(y, dim * 8, 5, 2)

    # fc: dim*8 -> 1
    logit = fc(y, 1)
    return logit


'''
  Discriminator model D (WGAN-GP)
  - Input img: image data from dataset or the G model
  - Output logit: the scalar to represent the prob that net belongs to the real data
'''
def discriminator_ln(img, dim=64, reuse=True, training=True):
  # standard convolutional layer using lrelu
  conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
  lrelu = partial(leak_relu, leak=0.2)

  # stacked layer: conv + ln + leaky relu  
  conv_ln_lrelu = partial(conv, normalizer_fn=slim.layer_norm, activation_fn=lrelu, biases_initializer=None)

  # fully connected layer
  fc = partial(flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # model D
  with tf.variable_scope('Model_D', reuse=reuse):
    # 1st conv: 1 -> dim
    y = lrelu(conv(img, dim, 5, 2))

    # 2nd conv: dim -> dim*2
    y = conv_ln_lrelu(y, dim * 2, 5, 2)

    # 3rd conv: dim*2 -> dim*4
    y = conv_ln_lrelu(y, dim * 4, 5, 2)

    # 4th conv: dim*4 -> dim*8
    y = conv_ln_lrelu(y, dim * 8, 5, 2)

    # fc: dim*8 -> 1
    logit = fc(y, 1)
    return logit


'''
  Encoder section for AutoEncoder
  - conv + bn + relu
'''
def encoder_AE(img, dim=4, reuse=True, training=True):
  # batch normalize layer
  batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
  bn = partial(batch_norm, is_training=training)

  # standard convolutional layer using relu  
  conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))

  # stacked layer: conv + bn + relu
  conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=tf.nn.relu, biases_initializer=None)

  # the encoder section
  with tf.variable_scope('encoder_AE', reuse=reuse):
    # 1st conv: 1 -> dim (1->4)
    y = tf.nn.relu((conv(img, dim, 3, 2)))

    # 2nd conv: dim -> dim*2 (4->8)
    y = conv_bn_relu(y, dim * 2, 3, 2)

    # 3rd conv: dim*2 -> dim*4 (8->16)
    y = conv_bn_relu(y, dim * 4, 3, 2)

    # 4th conv: dim*4 -> dim*8 (16->32)
    y = conv_bn_relu(y, dim * 8, 3, 2)

    # 5th conv: dim*8 -> dim*16 (32->64)
    y = conv_bn_relu(y, dim * 16, 3, 2)

    # output shape (N, 2, 2, 64)
    logit = y
    return logit


'''
  Decoder section for AutoEncoder
  - deconv + bn + relu
'''
def decoder_AE(code, dim=4, reuse=True, training=True):
  # batch normalize layer
  batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
  bn = partial(batch_norm, is_training=training)

  # deconvolutional layer
  dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

  # stacked layer: dconv + bn + relu
  dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=tf.nn.relu, biases_initializer=None)

  # the decoder section (reverse the encoder)
  with tf.variable_scope('decoder_AE', reuse=reuse):
    # 1st deconv: dim*16 -> dim*8 (64->32)
    y = dconv_bn_relu(code, dim * 8, 3, 2)

    # 2nd deconv: dim*8 -> dim*4 (32->16)
    y = dconv_bn_relu(y, dim * 4, 3, 2)

    # 3rd deconv: dim*4 -> dim*2 (16->8)
    y = dconv_bn_relu(y, dim * 2, 3, 2)

    # 4th deconv: dim*2 -> dim (8->4)
    y = dconv_bn_relu(y, dim, 3, 2)

    # 5th deconv: dim -> 1 (4->1)
    img = tf.tanh(dconv(y, 1, 3, 2))

    # return shape (N, 64, 64, 1)
    return img



