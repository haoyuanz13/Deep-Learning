from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
from sklearn.cluster import KMeans

from utils import *
from dataLoader import *
from model import *

slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

################
# Define flags #
################
flags = tf.app.flags
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir_cufs", 'ae_res/cufs_samples', "Directory to save cufs samples")
flags.DEFINE_string("sampledir_celeba", 'ae_res/celeba_samples', "Directory to save celeba samples")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("sample_size", 32, "The size of sample images [32]")
flags.DEFINE_integer("max_iterations", 10001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 10, "The step interval to plot training loss [1000]")
flags.DEFINE_integer("interval_save", 100, "The step interval to save generative images [1000]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS


##########################
# AutoEncoder main train #
##########################
def AutoEncoder_main(dataset, out_channel, cufs=True):
  # Models
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, out_channel])
  
  # encoder section
  code_AE = encoder_AE(x, reuse=False)

  # decoder section
  img_decoder = decoder_AE(code_AE, reuse=False)

  # generate sample images
  code_AE_sample = encoder_AE(x, reuse=True, training=False)
  img_decoder_sample = decoder_AE(code_AE_sample, reuse=True, training=False)

  # Optimizers
  t_vars = tf.trainable_variables()
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # L2 losses
  loss = l2_loss(img_decoder, x)
  # tf.summary.scalar('loss', loss)

  # optimizer for model training (.000002)
  # lr = .0002
  lr = .001

  ae_trainer = tf.train.AdamOptimizer(lr, beta1=.5).minimize(
      loss, global_step=global_step, var_list=[v for v in t_vars])

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  loss_set = np.zeros([int(FLAGS.max_iterations)])
  x_ipt_sample = dataset[:FLAGS.batch_size, :, :, :]

  # Training loop
  for step in range(2 if FLAGS.debug else int(FLAGS.max_iterations)): 
    # data x random shuffle 
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    data_step = dataset[arr[:], :, :, :]
    x_batch = data_step[0 : FLAGS.batch_size]

    # train Autoencoder
    _, loss_cur = sess.run([ae_trainer, loss], feed_dict={x: x_batch})

    # Log details
    if step % FLAGS.interval_plot == 0:
      print ('====> The {}th training step ||  L2 Loss: {:.8f}:'.format(step, loss_cur))

    loss_set[step] = loss_cur

    # Early stopping
    if np.isnan(loss_cur) or np.isnan(loss_cur):
      print('Early stopping')
      break

    # plot generative images
    if step % FLAGS.interval_save == 0:
      print ('\n===========> Generate sample images and saving ................................. \n')
      if FLAGS.sampledir_cufs or FLAGS.sampledir_celeba:
        images = sess.run(img_decoder_sample, feed_dict={x: x_ipt_sample})

        samples = FLAGS.sample_size
        images = images[:samples, :, :, :]
        images = np.reshape(images, [samples, 64, 64])
        images = (images + 1.) / 2.

        if cufs:
          scipy.misc.imsave(FLAGS.sampledir_cufs + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('ae_res/cufs_curve/loss.npy', loss_set[:step + 1])
        
        else:
          scipy.misc.imsave(FLAGS.sampledir_celeba + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('ae_res/celeba_curve/loss.npy', loss_set[:step + 1])

  return


########
# Main #
########
def main(cufs=True):
  # load data
  if cufs:
    dataset, _, _, _ = dataloader_cufs()
  else:
    dataset = dataloader_celeba(3)

  # obtain output channel number (e.g. 1 for grayscale or 3 for rgb)
  channel = dataset.shape[-1]
  AutoEncoder_main(dataset, channel, cufs)


if __name__ == '__main__':
  main(cufs=False)

  # processPlot()
