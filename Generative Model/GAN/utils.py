import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1]))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j * h:j * h + h, i * w:i * w + w] = image

  return img


# leaky relu function
def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


# fully connected layer
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
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
def batch_norm(x, momentum, epsilon, name, train=True):
  return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None,
                      epsilon=epsilon, scale=True, is_training=train, scope=name)



'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_loss(iteration, d_loss, g_loss):
  fig, (Ax0, Ax1) = plt.subplots(2, 1, figsize = (8, 20))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, d_loss)
  Ax0.set_title('Model D Loss vs Iterations') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('loss value')
  Ax0.grid(True)
  

  Ax1.plot(x, g_loss)
  Ax1.set_title('Model G Loss vs Iterations')
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('loss value')
  Ax1.grid(True)

  plt.show()


'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot():
  loss_d = np.load('res/loss_modelD.npy')
  loss_g = np.load('res/loss_modelG.npy')

  total = loss_d.shape[0]

  processPlot_loss(total, loss_d, loss_g)
