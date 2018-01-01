# from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
# import cv2
from PIL import Image
import pdb
import imageio, os

from functools import partial

flags = tf.app.flags
flags.DEFINE_string("dcgan_sampledir_cufs", 'dcgan_res/cufs_samples', "Directory to save cufs samples (dcgan)")
flags.DEFINE_string("dcgan_sampledir_celeba", 'dcgan_res/celeba_samples', "Directory to save celeba samples (dcgan)")
flags.DEFINE_string("dcgan_curve_cufs", 'dcgan_res/cufs_curve', "Directory to save cufs loss curve (dcgan)")
flags.DEFINE_string("dcgan_curve_celeba", 'dcgan_res/celeba_curve', "Directory to save celeba loss curve (dcgan)")

flags.DEFINE_string("wgan_gp_sampledir_cufs", 'wgan_gp_res/cufs_samples', "Directory to save cufs samples (wgan_gp)")
flags.DEFINE_string("wgan_gp_sampledir_celeba", 'wgan_gp_res/celeba_samples', "Directory to save celeba samples (wgan_gp)")
flags.DEFINE_string("wgan_gp_curve_cufs", 'wgan_gp_res/cufs_curve', "Directory to save cufs loss curve (wgan_gp)")
flags.DEFINE_string("wgan_gp_curve_celeba", 'wgan_gp_res/celeba_curve', "Directory to save celeba loss curve (wgan_gp)")

flags.DEFINE_string("wgan_sampledir_cufs", 'wgan_res/cufs_samples', "Directory to save cufs samples (wgan)")
flags.DEFINE_string("wgan_sampledir_celeba", 'wgan_res/celeba_samples', "Directory to save celeba samples (wgan)")
flags.DEFINE_string("wgan_curve_cufs", 'wgan_res/cufs_curve', "Directory to save cufs loss curve (wgan)")
flags.DEFINE_string("wgan_curve_celeba", 'wgan_res/celeba_curve', "Directory to save celeba loss curve (wgan)")

flags.DEFINE_string("ae_sampledir_cufs", 'ae_res/cufs_samples', "Directory to save cufs samples (ae)")
flags.DEFINE_string("ae_sampledir_celeba", 'ae_res/celeba_samples', "Directory to save celeba samples (ae)")
flags.DEFINE_string("ae_curve_cufs", 'ae_res/cufs_curve', "Directory to save cufs loss curve (ae)")
flags.DEFINE_string("ae_curve_celeba", 'ae_res/celeba_curve', "Directory to save celeba loss curve (ae)")

flags.DEFINE_string("vae_sampledir_cufs", 'vae_res/cufs_samples', "Directory to save cufs samples (vae)")
flags.DEFINE_string("vae_sampledir_celeba", 'vae_res/celeba_samples', "Directory to save celeba samples (vae)")
flags.DEFINE_string("vae_curve_cufs", 'vae_res/cufs_curve', "Directory to save cufs loss curve (vae)")
flags.DEFINE_string("vae_curve_celeba", 'vae_res/celeba_curve', "Directory to save celeba loss curve (vae)")
FLAGS = flags.FLAGS

'''
  Add noise
'''
def addNoise(x):
  x = x * np.random.randint(2, size=x.shape)
  x += np.random.randint(2, size=x.shape)

  return x



def merge(images, size):
  h, w, c = images.shape[1], images.shape[2], images.shape[-1]
  img = np.zeros((h * size[0], w * size[1]))

  N = size[0] * size[1]
  for idx, image in enumerate(images):
    if idx >= N:
      break

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
  helper function for the gradient penalty
'''
def interpolate(a, b):
  shape = tf.concat((tf.shape(a)[0 : 1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
  alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
  inter = a + alpha * (b - a)
  inter.set_shape(a.get_shape().as_list())
  return inter


'''
  gradient penalty for WGAN
'''
def gradient_penalty(real, fake, fun):
  x = interpolate(real, fake)
  pred = fun(x, reuse=True)
  gradients = tf.gradients(pred, x)[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
  gp = tf.reduce_mean((slopes - 1.)**2)
  return gp

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
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_loss_GANs(path, iteration, d_loss, g_loss):
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
  
  fig.savefig(path + '/loss_curve.png')   # save the figure to file
  plt.close(fig)    # close the figure


'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot_GANs(ganType=0, cufs=True):
  if ganType == 0:
    print ("======> Loss curve generated from DCGAN model")
    path = FLAGS.dcgan_curve_cufs if cufs else FLAGS.dcgan_curve_celeba

  if ganType == 1:
    print ("======> Loss curve generated from WGAN model")
    path = FLAGS.wgan_curve_cufs if cufs else FLAGS.wgan_curve_celeba

  if ganType == 2:
    print ("======> Loss curve generated from WGAN-GP model")
    path = FLAGS.wgan_gp_curve_cufs if cufs else FLAGS.wgan_gp_curve_celeba

  loss_d = np.load(path + '/loss_modelD.npy')
  loss_g = np.load(path + '/loss_modelG.npy')

  total = loss_d.shape[0]

  processPlot_loss_GANs(path, total, loss_d, loss_g)


'''
  generate gif
'''
def gifGenerate_GANs(ganType=0, cufs=True):
  if ganType == 0:
    print ("======> Samples generated from DCGAN model")
    path = FLAGS.dcgan_sampledir_cufs if cufs else FLAGS.dcgan_sampledir_celeba

  if ganType == 1:
    print ("======> Samples generated from WGAN model")
    path = FLAGS.wgan_sampledir_cufs if cufs else FLAGS.wgan_sampledir_celeba

  if ganType == 2:
    print ("======> Samples generated from WGAN-GP model")
    path = FLAGS.wgan_gp_sampledir_cufs if cufs else FLAGS.wgan_gp_sampledir_celeba

  imgs = []
  print (path)
  for filename in sorted(os.listdir(path)):
    print (filename)
    imgs.append(imageio.imread(path + '/' + filename))

  imageio.mimsave(path + '/samples.gif', imgs)




'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_loss_AEs(path, iteration, loss):
  fig, (Ax0) = plt.subplots(1, 1, figsize = (8, 20))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, loss)
  Ax0.set_title('Training Loss vs Iterations') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('loss value')
  Ax0.grid(True)

  plt.show()
  
  fig.savefig(path + '/loss_curve.png')   # save the figure to file
  plt.close(fig)    # close the figure



'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot_AEs(cufs=True):
  path = FLAGS.ae_curve_cufs if cufs else FLAGS.ae_curve_celeba
  loss = np.load(path + '/loss.npy')
  total = loss.shape[0]

  processPlot_loss_AEs(path, total, loss)


'''
  generate gif
'''
def gifGenerate_AEs(aeType=0, cufs=True):
  if aeType == 0:
    print ("======> Samples generated from AutoEncoder model")
    path = FLAGS.ae_sampledir_cufs if cufs else FLAGS.ae_sampledir_celeba

  if aeType == 1:
    print ("======> Samples generated from VAE model")
    path = FLAGS.vae_sampledir_cufs if cufs else FLAGS.vae_sampledir_celeba

  imgs = []
  print (path)
  for filename in sorted(os.listdir(path)):
    print (filename)
    imgs.append(imageio.imread(path + '/' + filename))

  imageio.mimsave(path + '/samples.gif', imgs)




'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_loss_VAEs(path, iteration, loss_ML, loss_KL):
  fig, (Ax0, Ax1) = plt.subplots(2, 1, figsize = (8, 20))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, loss_ML)
  Ax0.set_title('Marginal Likelihood vs Iterations') 
  # Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('ML value')
  Ax0.grid(True)

  Ax1.plot(x, loss_KL)
  Ax1.set_title('KL Divergence vs Iterations')
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('KL value')
  Ax1.grid(True)

  plt.show()
  fig.savefig(path + '/loss_curve.png')   # save the figure to file
  plt.close(fig)    # close the figure



'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot_VAEs(cufs=True):
  path = FLAGS.vae_curve_cufs if cufs else FLAGS.vae_curve_celeba

  # load loss arraies
  loss_ML = np.load(path + '/loss_ML.npy')
  loss_KL = np.load(path + '/loss_KL.npy')

  total = loss_ML.shape[0]

  processPlot_loss_VAEs(path, total, loss_ML, loss_KL)




if __name__ == "__main__":
  '''
    GAN Type
    - 0: DCGAN
    - 1: WGAN
    - 2: WGAN-GP
  '''
  # processPlot_GANs(ganType=1, cufs=False)
  # gifGenerate_GANs(ganType=1, cufs=False)

  # processPlot_AEs(cufs=False)
  processPlot_VAEs(cufs=True)
  gifGenerate_AEs(aeType=1, cufs=True)




