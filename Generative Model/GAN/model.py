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

slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

################
# Define flags #
################

flags = tf.app.flags
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir", 'res', "Directory to save samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("train_steps_model_D", 2, "The training iterations of model D [2]")
flags.DEFINE_integer("max_iterations", 1e6, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 1000, "The step interval to plot generative images [1000]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS

###############
# DCGAN Model #
###############

'''
  Generative Model G
  - Input z: random noise vector
  - Output net: generated map with same size as data
'''
def generator(z):
  # the output matrix size after reshape
  init_height, init_width = 4, 4
  channel_num = (1024, 512, 256, 128, 1)

  with tf.variable_scope("Model_G") as scope:
    # fc converts noise vector z into required size 
    net = linear(z, init_height * init_width * channel_num[0], 'g_fc')
    # reshape feature vector into matrix with size [bs, height, width, channel]
    net = tf.reshape(net, [-1, init_height, init_width, channel_num[0]])
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn0'))

    # 1st deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 8, 8, channel_num[1]], name='g_deconv1')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn1'))    

    # 2nd deconvolution (keep same size)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 8, 8, channel_num[1]], name='g_deconv2', d_h=1, d_w=1)
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn2'))    

    # 3rd deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 16, 16, channel_num[2]], name='g_deconv3')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn3'))   

    # 4th deconvolution (keep same size)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 16, 16, channel_num[2]], name='g_deconv4', d_h=1, d_w=1)
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn4'))     

    # 5th deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 32, 32, channel_num[3]], name='g_deconv5')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn5'))    

    # 6th deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 64, 64, channel_num[-1]], name='g_deconv6')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn6'))    

    # nonlinearize
    net = tf.nn.tanh(net)

    # tf.histogram_summary('Model_G/out', net)
    tf.summary.histogram('Model_G/out', net)
    # tf.image_summary("Model_G", net, max_images=8)

  return net

'''
  Discriminator model D
  - Input net: image data from dataset or the G model
  - Output prob: the scalar to represent the prob that net belongs to the real data
'''
def discriminator(net, reuse=False):
  with tf.variable_scope("Model_D") as scope:
    if reuse:
      scope.reuse_variables()

    # simulate the inverse operation of the generative model G
    channel_num = (128, 256, 512, 1024)

    # standard convolutional layer
    # 1st convolutional layer
    feaMap = lrelu(conv2d(net, output_dim=channel_num[0], name='d_conv0'))

    # 2nd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[1], name='d_conv1')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn1'))

    # 3rd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[1], name='d_conv2', d_h=1, d_w=1)
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn2'))

    # 4th convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[2], name='d_conv3')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn3'))

    # 5th convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[2], name='d_conv4', d_h=1, d_w=1)
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn4'))

    # 6th convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[3], name='d_conv5')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn5'))

    # reshape feature map and use fc to get 1 size output as prob
    prob = linear(tf.reshape(feaMap, [FLAGS.batch_size, -1]), 1, 'd_fc')

    # apply sigmoid for prob computation
    return tf.nn.sigmoid(prob)
   


####################
# DCGAN main train #
####################
def DCGAN_main(dataset):
  # Models
  z_dim = 100
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, 1])
  d_model = discriminator(x, reuse=False)

  z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])  
  g_model = generator(z)
  dg_model = discriminator(g_model, reuse=True)



  # Optimizers
  t_vars = tf.trainable_variables()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model))
  tf.summary.scalar('d_loss', d_loss)

  # optimizer for model D training
  lr_modelD = .000002
  lr_modelG = 15 * .000002

  d_trainer = tf.train.AdamOptimizer(lr_modelD, beta1=.5).minimize(
      d_loss, global_step=global_step, var_list=[v for v in t_vars if 'Model_D/' in v.name])

  # optimizer for model G training
  g_loss = tf.reduce_mean(1 - tf.log(dg_model))
  # g_loss = tf.reduce_mean(1. - tf.log(dg_model_train_g))
  tf.summary.scalar('g_loss', g_loss)
  g_trainer = tf.train.AdamOptimizer(lr_modelG, beta1=.5).minimize(
      g_loss, var_list=[v for v in t_vars if 'Model_G/' in v.name])

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # generate z noise
  z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)

  d_loss_set,  g_loss_set = np.zeros([int(FLAGS.max_iterations)]), np.zeros([int(FLAGS.max_iterations)])
  # Training loop
  for step in range(2 if FLAGS.debug else int(FLAGS.max_iterations)):    
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
    # sess.run(g_trainer, feed_dict={z: z_batch})
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
    if step != 0 and step % FLAGS.interval_plot == 0:

      print ('\n===========> Generate sample images and saving ................................. \n')
      if FLAGS.sampledir:
        samples = FLAGS.batch_size
        images = sess.run(g_model, feed_dict={z: z_batch})
        images = np.reshape(images, [samples, 64, 64])
        images = (images + 1.) / 2.
        scipy.misc.imsave(FLAGS.sampledir + ('/new_sample_{}.png'.format(step)), merge(images, [int(math.sqrt(samples))] * 2))

        np.save('res/loss_modelD.npy', d_loss_set[:step + 1])
        np.save('res/loss_modelG.npy', g_loss_set[:step + 1])

  return


########
# Main #
########
def main():
  # load data
  train_dict_cufs = dataProcess()
  test_dict_cufs = dataProcess(True)

  train_imgs_cufs, train_ind_cufs = train_dict_cufs['img'], train_dict_cufs['order']
  test_imgs_cufs, test_ind_cufs = test_dict_cufs['img'], test_dict_cufs['order']

  DCGAN_main(train_imgs_cufs)


if __name__ == '__main__':
  main()

  # processPlot()
