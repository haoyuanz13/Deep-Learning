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
flags.DEFINE_string("sampledir_cufs", 'dcgan_res/cufs_samples', "Directory to save cufs samples")
flags.DEFINE_string("sampledir_celeba", 'dcgan_res/celeba_samples', "Directory to save celeba samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("sample_size", 32, "The size of sample images [32]")
flags.DEFINE_integer("train_steps_model_D", 1, "The training iterations of model D [2]")
flags.DEFINE_integer("max_iterations", 5001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 50, "The step interval to plot generative images [1000]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS


####################
# DCGAN main train #
####################
def DCGAN_main(dataset, out_channel, cufs=True):
  # Models
  z_dim = 100
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, out_channel])
  d_model = discriminator_bn(x, reuse=False)

  z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])  
  g_model = generator(z, reuse=False)
  dg_model = discriminator_bn(g_model, reuse=True)

  f_sample = generator(z, reuse=True, training=False)

  # Optimizers
  t_vars = tf.trainable_variables()
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # losses
  d_r_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_model), d_model)
  d_f_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(dg_model), dg_model)

  d_loss = (d_r_loss + d_f_loss) / 2.0
  tf.summary.scalar('d_loss', d_loss)

  # optimizer for model D training (.000002)
  lr_modelD = .0002
  lr_modelG = .0002

  d_trainer = tf.train.AdamOptimizer(lr_modelD, beta1=.5).minimize(
      d_loss, global_step=global_step, var_list=[v for v in t_vars if 'Model_D/' in v.name])

  # optimizer for model G training
  g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(dg_model), dg_model)
  tf.summary.scalar('g_loss', g_loss)
  g_trainer = tf.train.AdamOptimizer(lr_modelG, beta1=.5).minimize(
      g_loss, var_list=[v for v in t_vars if 'Model_G/' in v.name])

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  d_loss_set,  g_loss_set = np.zeros([int(FLAGS.max_iterations)]), np.zeros([int(FLAGS.max_iterations)])
  
  # Training loop
  z_ipt_sample = np.random.normal(size=[FLAGS.batch_size, z_dim]).astype(np.float32)
  for step in range(2 if FLAGS.debug else int(FLAGS.max_iterations)): 
    # generate z noise
    # z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
    z_batch = np.random.normal(size=[FLAGS.batch_size, z_dim]).astype(np.float32)

    # data x random shuffle 
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    data_step = dataset[arr[:], :, :, :]
    x_batch = data_step[0 : FLAGS.batch_size]

    # update model D for k times
    d_loss_val = 0
    for k in range(FLAGS.train_steps_model_D):
      # Update discriminator
      _, d_loss_val_cur = sess.run([d_trainer, d_loss], feed_dict={x: x_batch, z: z_batch})
      d_loss_val += d_loss_val_cur

    d_loss_val /= FLAGS.train_steps_model_D

    # update model G for one time 
    sess.run(g_trainer, feed_dict={z: z_batch})
    _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z: z_batch})


    # Log details
    print ('====> The {}th training step || Model G Loss: {:.8f} || Model D Loss: {:.8f}'.format(step, g_loss_val, d_loss_val))

    d_loss_set[step], g_loss_set[step] = d_loss_val, g_loss_val
    # summary_str = sess.run(summary, feed_dict={x: images, z: z_batch})
    # summary_writer.add_summary(summary_str, global_step.eval())

    # Early stopping
    if np.isnan(g_loss_val) or np.isnan(g_loss_val):
      print('Early stopping')
      break


    # plot generative images
    if step % FLAGS.interval_plot == 0:

      print ('\n===========> Generate sample images and saving ................................. \n')
      if FLAGS.sampledir_cufs or FLAGS.sampledir_celeba:
        samples = FLAGS.sample_size
        images = sess.run(f_sample, feed_dict={z: z_ipt_sample})

        images = images[:samples, :, :, :]
        images = np.reshape(images, [samples, 64, 64])
        images = (images + 1.) / 2.

        if cufs:
          scipy.misc.imsave(FLAGS.sampledir_cufs + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('dcgan_res/cufs_curve/loss_modelD.npy', d_loss_set[:step + 1])
          np.save('dcgan_res/cufs_curve/loss_modelG.npy', g_loss_set[:step + 1])
        
        else:
          scipy.misc.imsave(FLAGS.sampledir_celeba + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('dcgan_res/celeba_curve/loss_modelD.npy', d_loss_set[:step + 1])
          np.save('dcgan_res/celeba_curve/loss_modelG.npy', g_loss_set[:step + 1])

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
  DCGAN_main(dataset, channel, cufs)


if __name__ == '__main__':
  main()

  # processPlot()
