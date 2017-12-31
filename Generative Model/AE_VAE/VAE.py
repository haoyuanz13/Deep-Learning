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
flags.DEFINE_string("sampledir_cufs", 'vae_res/cufs_samples', "Directory to save cufs samples")
flags.DEFINE_string("sampledir_celeba", 'vae_res/celeba_samples', "Directory to save celeba samples")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [128]")
flags.DEFINE_integer("sample_size", 32, "The size of sample images [32]")
flags.DEFINE_integer("max_iterations", 10001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 10, "The step interval to plot training loss [1000]")
flags.DEFINE_integer("interval_save", 100, "The step interval to save generative images [1000]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS

#############
# VAE model #
#############
def VAE(x, reuse=False, is_train=True):
  with tf.variable_scope('VAE', reuse=reuse) as vae:
    with tf.variable_scope('VAE_feed', reuse=reuse):
      # generate mu and std via encoding
      mu, std = encoder_VAE(x, reuse=reuse, training=is_train)

      # sample mu and std to get code z
      z = mu + std * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

      # decoding
      f_z = decoder_VAE(z, reuse=reuse, training=is_train)


  # marginal likelihood (negative value)
  marginal_likelihood = tf.reduce_mean(tf.reduce_sum(
                        x * tf.log(f_z) + (1 - x) * tf.log(1 - f_z), 1))  

  # KL divergence (positive value)
  KL_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(
                  tf.square(mu) + tf.square(std) - tf.log(tf.square(std)) - 1, 1))

  # equal weight comb loss    
  comb_loss = KL_divergence - marginal_likelihood

  # variables
  var_vae = tf.contrib.framework.get_variables(vae)

  return f_z, comb_loss, -marginal_likelihood, KL_divergence, var_vae



##################
# VAE main train #
##################
def VAE_main(dataset, out_channel, cufs=True):
  # Models
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, out_channel])

  # VAE feed forward
  f_z, comb_loss, ml_train, kl_train, var_vae = VAE(x, reuse=False, is_train=True)

  # sample generation
  f_z_samples, _, _, _, _ = VAE(x, reuse=True, is_train=False)  
  
  # VAE trainer 
  lr_vae = .005  #.0002
  global_step = tf.Variable(0, name='global_step', trainable=False)
  vae_trainer = tf.train.AdamOptimizer(lr_vae, beta1=.5).minimize(comb_loss, global_step=global_step, var_list=var_vae)

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  Comb_loss_set = np.zeros([int(FLAGS.max_iterations)])
  ML_loss_set,  KL_loss_set = np.zeros([int(FLAGS.max_iterations)]), np.zeros([int(FLAGS.max_iterations)])
  
  # Training loop
  x_ipt_sample = dataset[:FLAGS.batch_size, :, :, :]
  for step in range(2 if FLAGS.debug else int(FLAGS.max_iterations)): 
    # data x random shuffle 
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    data_step = dataset[arr[:], :, :, :]
    x_batch = data_step[0 : FLAGS.batch_size]

    # train the model and eval losses
    _, train_comb_loss, train_ML_loss, train_KL_div = \
        sess.run([vae_trainer, comb_loss, ml_train, kl_train], feed_dict={x: x_batch})

    # store training loss
    Comb_loss_set[step], ML_loss_set[step], KL_loss_set[step] = train_comb_loss, train_ML_loss, train_KL_div

    # Log details
    if step % FLAGS.interval_plot == 0:
      print ('====> The {}th training step || Comb Loss {:.8f} || ML Likelihood: {:.8f} || KL Divergence: {:.8f}'.format(
        step, train_comb_loss, train_ML_loss, train_KL_div))

    # Early stopping
    if np.isnan(train_comb_loss) or np.isnan(train_ML_loss) or np.isnan(train_KL_div):
      print('Early stopping')
      break

    # plot generative images
    if step % FLAGS.interval_save == 0:

      print ('\n===========> Generate sample images and saving ................................. \n')
      if FLAGS.sampledir_cufs or FLAGS.sampledir_celeba:
        samples = FLAGS.sample_size
        images = sess.run(f_z_samples, feed_dict={x: x_ipt_sample})

        images = images[:samples, :, :, :]
        images = np.reshape(images, [samples, 64, 64])
        images = (images + 1.) / 2.

        if cufs:
          scipy.misc.imsave(FLAGS.sampledir_cufs + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('vae_res/cufs_curve/loss_comb.npy', Comb_loss_set[:step + 1])
          np.save('vae_res/cufs_curve/loss_ML.npy', ML_loss_set[:step + 1])
          np.save('vae_res/cufs_curve/loss_KL.npy', KL_loss_set[:step + 1])
        
        else:
          scipy.misc.imsave(FLAGS.sampledir_celeba + ('/sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('vae_res/celeba_curve/loss_comb.npy', Comb_loss_set[:step + 1])
          np.save('vae_res/celeba_curve/loss_ML.npy', ML_loss_set[:step + 1])
          np.save('vae_res/celeba_curve/loss_KL.npy', KL_loss_set[:step + 1])

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
  VAE_main(dataset, channel, cufs)


if __name__ == '__main__':
  main()

  # processPlot()
