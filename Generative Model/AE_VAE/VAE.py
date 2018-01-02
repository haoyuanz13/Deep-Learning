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
flags.DEFINE_integer("total_num", 88, "The size of the dataset [88]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [128]")
flags.DEFINE_integer("sample_size", 32, "The size of sample images [32]")
flags.DEFINE_integer("out_channel", 1, "The channel number of data [1]")
flags.DEFINE_integer("max_iterations", 10001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 10, "The epoch interval to plot training loss [1000]")
flags.DEFINE_integer("interval_save", 100, "The epoch interval to save generative images [1000]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
flags.DEFINE_boolean("addNoise", True, "True if adding noise to the raw data")
FLAGS = flags.FLAGS



#############
# VAE model #
#############
def VAE(x_hat, x, cufs=True, reuse=False, is_train=True):
  with tf.variable_scope('VAE', reuse=reuse) as vae:
    with tf.variable_scope('VAE_feed', reuse=reuse):
      # generate mu and std via encoding
      if cufs:
        mu, std = encoder_VAE(x_hat, reuse=reuse, training=is_train)
      else:
        mu, std = encoder_largerVAE(x_hat, reuse=reuse, training=is_train)

      # sample mu and std to get code z
      z = mu + std * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

      # decoding
      if cufs:
        f_z = decoder_VAE(z, reuse=reuse, training=is_train)
      else:
        f_z = decoder_largerVAE(z, reuse=reuse, training=is_train)

  # marginal likelihood (negative value)
  marginal_likelihood = tf.reduce_mean(tf.reduce_sum(
                        x * tf.log(f_z) + (1 - x) * tf.log(1 - f_z), 1))  

  # KL divergence (positive value)
  KL_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(
                  tf.square(mu) + tf.square(std) - tf.log(tf.square(std)) - 1, 1))

  KL_divergence *= 0.01

  # equal weight comb loss    
  comb_loss = KL_divergence - marginal_likelihood

  # variables
  var_vae = tf.contrib.framework.get_variables(vae)

  return f_z, comb_loss, -marginal_likelihood, KL_divergence, var_vae



##################
# VAE main train #
##################
def VAE_main(dataset_train, dataset_test, cufs=True):
  # x represents the input raw data
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, FLAGS.out_channel], name='target_img')
  # if addNoise: x_hat = x+noise; else: x_hat = x
  x_hat = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, FLAGS.out_channel], name='input_img')


  # VAE feed forward
  f_z, comb_loss, ml_train, kl_train, var_vae = VAE(x_hat, x, cufs, reuse=False, is_train=True)

  # sample generation
  f_z_samples, _, _, _, _ = VAE(x_hat, x, cufs, reuse=True, is_train=False)  
  
  # VAE trainer 
  lr_vae = .001
  global_step = tf.Variable(0, name='global_step', trainable=False)
  vae_trainer = tf.train.AdamOptimizer(lr_vae, beta1=.5).minimize(comb_loss, global_step=global_step, var_list=var_vae)

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  Comb_loss_set = np.zeros([int(FLAGS.max_iterations)])
  ML_loss_set,  KL_loss_set = np.zeros([int(FLAGS.max_iterations)]), np.zeros([int(FLAGS.max_iterations)])
  
  # test data for reconstruction
  x_ipt_sample = dataset_test[:FLAGS.batch_size, :, :, :]
  x_ipt_sample_target = x_ipt_sample
  x_ipt_sample_input = addNoise(x_ipt_sample) if FLAGS.addNoise else x_ipt_sample

  # save original and input data
  test_ori = merge(x_ipt_sample_target[:, :, :, 0], [int(math.sqrt(FLAGS.sample_size))] * 2)
  test_input = merge(x_ipt_sample_input[:, :, :, 0], [int(math.sqrt(FLAGS.sample_size))] * 2)

  save_path = FLAGS.sampledir_cufs if cufs else FLAGS.sampledir_celeba
  scipy.misc.imsave(save_path + '/original_input.png', test_ori)
  scipy.misc.imsave(save_path + '/actual_input.png', test_input)

  # Training loop
  for epoch in range(2 if FLAGS.debug else int(FLAGS.max_iterations)): 
    # data x random shuffle 
    arr = np.arange(FLAGS.total_num)
    np.random.shuffle(arr)
    data_epoch = dataset_train[arr[:], :, :, :]

    total_step = FLAGS.total_num / FLAGS.batch_size
    # total_step = 1

    train_comb_loss, train_ML_loss, train_KL_div = 0, 0, 0
    for step in range(total_step):
      x_batch = data_epoch[step * FLAGS.batch_size : (step + 1) * FLAGS.batch_size]
      # x_batch = data_epoch[0 : FLAGS.batch_size]

      x_target = x_batch  # raw data
      x_input = addNoise(x_batch) if FLAGS.addNoise else x_batch  # add noise if necessary

      # train the model and eval losses
      _, train_comb_loss_step, train_ML_loss_step, train_KL_div_step = \
          sess.run([vae_trainer, comb_loss, ml_train, kl_train], feed_dict={x_hat: x_input, x: x_target})

      # update loss sum
      train_comb_loss += train_comb_loss_step
      train_ML_loss += train_ML_loss_step
      train_KL_div += train_KL_div_step


    # store training loss
    Comb_loss_set[epoch], ML_loss_set[epoch], KL_loss_set[epoch] = train_comb_loss/total_step, train_ML_loss/total_step, train_KL_div/total_step

    # Log details
    if epoch % FLAGS.interval_plot == 0:
      print ('====> The {}th training epoch || Comb Loss {:.8f} || ML Likelihood: {:.8f} || KL Divergence: {:.8f}'.format(
        epoch, train_comb_loss/total_step, train_ML_loss/total_step, train_KL_div/total_step))

    # Early stopping
    if np.isnan(train_comb_loss) or np.isnan(train_ML_loss) or np.isnan(train_KL_div):
      print('Early stopping')
      break

    # plot generative images
    if epoch % FLAGS.interval_save == 0:
      print ('\n===========> Generate sample images and saving ................................. \n')
      if FLAGS.sampledir_cufs or FLAGS.sampledir_celeba:
        samples = FLAGS.sample_size
        images = sess.run(f_z_samples, feed_dict={x_hat: x_ipt_sample_input, x: x_ipt_sample_target})

        images = images[:samples, :, :, :]
        images = np.reshape(images, [samples, 64, 64])
        images = (images + 1.) / 2.

        if cufs:
          scipy.misc.imsave(FLAGS.sampledir_cufs + ('/sample_{}.png'.format(epoch)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('vae_res/cufs_curve/loss_comb.npy', Comb_loss_set[:epoch + 1])
          np.save('vae_res/cufs_curve/loss_ML.npy', ML_loss_set[:epoch + 1])
          np.save('vae_res/cufs_curve/loss_KL.npy', KL_loss_set[:epoch + 1])
        
        else:
          scipy.misc.imsave(FLAGS.sampledir_celeba + ('/sample_{}.png'.format(epoch)), merge(images, [int(math.sqrt(samples))] * 2))
          np.save('vae_res/celeba_curve/loss_comb.npy', Comb_loss_set[:epoch + 1])
          np.save('vae_res/celeba_curve/loss_ML.npy', ML_loss_set[:epoch + 1])
          np.save('vae_res/celeba_curve/loss_KL.npy', KL_loss_set[:epoch + 1])

  return


########
# Main #
########
def main(cufs=True):
  # load data
  if cufs:
    dataset_train, _, dataset_test, _ = dataloader_cufs()
  else:
    dataset_train = dataloader_celeba(3, twoSides=False)
    dataset_test = dataloader_celeba(1, twoSides=False)

  # obtain output channel number (e.g. 1 for grayscale or 3 for rgb)
  FLAGS.total_num = dataset_train.shape[0]
  FLAGS.out_channel = dataset_train.shape[-1]
  
  VAE_main(dataset_train, dataset_test, cufs)


if __name__ == '__main__':
  main(cufs=False)

  # processPlot()
