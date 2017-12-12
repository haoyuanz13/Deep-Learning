from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
# from sklearn.cluster import KMeans

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
flags.DEFINE_string("sampledir_cufs", 'wgan_res/cufs_samples', "Directory to save cufs samples")
flags.DEFINE_string("sampledir_celeba", 'wgan_res/celeba_samples', "Directory to save celeba samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("sample_size", 32, "The size of sample images [32]")
flags.DEFINE_integer("n_critic", 5, "The training iterations of model D [5]")
flags.DEFINE_integer("max_iterations", 5001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 50, "The step interval to plot generative images")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS


######################
# WGAN main train #
######################
def WGAN_main(dataset, out_channel, cufs=True):
  # Models
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, out_channel])
  real_prob = discriminator_bn(x, reuse=False)

  z_dim = 100
  z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])  
  fake_x = generator(z, reuse=False)
  fake_prob = discriminator_bn(fake_x, reuse=True)

  # max real prob (negative sign) and min fake prob (positive sign)
  wp = tf.reduce_mean(fake_prob) - tf.reduce_mean(real_prob)
  
  f_sample = generator(z, reuse=True, training=False)

  # Optimizers
  t_vars = tf.trainable_variables()
  # global_step = tf.Variable(0, name='global_step', trainable=False)

  # optimizer for model D training
  d_loss = wp
  tf.summary.scalar('d_loss', d_loss)
  lr_modelD = 0.0025 
  d_trainer = tf.train.RMSPropOptimizer(learning_rate=lr_modelD).minimize(
    d_loss, var_list=[v for v in t_vars if 'Model_D/' in v.name])

  # weight clipping
  clip = 0.005
  with tf.control_dependencies([d_trainer]):
    d_step = tf.group(*(tf.assign(var, tf.clip_by_value(var, -clip, clip)) for var in [v for v in t_vars if 'Model_D/' in v.name]))
  

  # optimizer for model G training
  g_loss = -tf.reduce_mean(fake_prob)
  tf.summary.scalar('g_loss', g_loss)
  lr_modelG = 0.0025
  g_step = tf.train.RMSPropOptimizer(learning_rate=lr_modelG).minimize(
    g_loss, var_list=[v for v in t_vars if 'Model_G/' in v.name])

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())


  # Training loop
  d_loss_set,  g_loss_set = np.zeros([int(FLAGS.max_iterations)]), np.zeros([int(FLAGS.max_iterations)])
  z_ipt_sample = np.random.normal(size=[FLAGS.batch_size, z_dim]).astype(np.float32)
  for step in range(2 if FLAGS.debug else int(FLAGS.max_iterations)): 
    # update model D for k times
    d_loss_val = 0
    for t in range(FLAGS.n_critic):
      # generate z noise
      # z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
      z_batch = np.random.normal(size=[FLAGS.batch_size, z_dim]).astype(np.float32)

      # data x random shuffle 
      arr = np.arange(dataset.shape[0])
      np.random.shuffle(arr)
      data_step = dataset[arr[:], :, :, :]
      x_batch = data_step[0 : FLAGS.batch_size]

      _, d_loss_val_cur = sess.run([d_step, d_loss], feed_dict={x: x_batch, z: z_batch})
      d_loss_val += d_loss_val_cur

    d_loss_val /= FLAGS.n_critic

    # update model G for one time 
    # sess.run(g_trainer, feed_dict={z: z_batch})
    z_batch = np.random.normal(size=[FLAGS.batch_size, z_dim]).astype(np.float32)
    _, g_loss_val = sess.run([g_step, g_loss], feed_dict={z: z_batch})


    # Log details
    d_loss_set[step], g_loss_set[step] = d_loss_val, g_loss_val
    print ('====> The {}th training step || Model G Loss: {:.8f} || Model D Loss: {:.8f}'.format(step, g_loss_val, d_loss_val))
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
          np.save('wgan_res/cufs_curve/loss_modelD.npy', d_loss_set[:step + 1])
          np.save('wgan_res/cufs_curve/loss_modelG.npy', g_loss_set[:step + 1])
        
        else:
          scipy.misc.imsave(FLAGS.sampledir_celeba + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('wgan_res/celeba_curve/loss_modelD.npy', d_loss_set[:step + 1])
          np.save('wgan_res/celeba_curve/loss_modelG.npy', g_loss_set[:step + 1])

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
  WGAN_main(dataset, channel, cufs)


if __name__ == '__main__':
  main()

  # processPlot()
