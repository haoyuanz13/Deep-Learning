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
flags.DEFINE_string("sampledir", None, "Directory to save samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("train_steps_model_D", 3, "The training iterations of model D [3]")
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
def generator(z, reuse=False):
  # the output matrix size after reshape
  init_height, init_width = 4, 4
  channel_num = (1024, 512, 256, 128, 1)

  with tf.variable_scope("Model_G") as scope:
    if reuse:
      scope.reuse_variables()

    # fc converts noise vector z into required size 
    net = linear(z, init_height * init_width * channel_num[0], 'g_fc')
    # reshape feature vector into matrix with size [bs, height, width, channel]
    net = tf.reshape(net, [-1, init_height, init_width, channel_num[0]])
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn0'))

    # 1st deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 8, 8, channel_num[1]], name='g_deconv1')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn1'))    

    # 2nd deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 16, 16, channel_num[2]], name='g_deconv2')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn2'))    

    # 3rd deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 32, 32, channel_num[3]], name='g_deconv3')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn3'))    

    # 4th deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 64, 64, channel_num[-1]], name='g_deconv4')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn4'))    

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

    #2nd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[1], name='d_conv1')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn1'))

    # 3rd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[2], name='d_conv2')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn2'))

    # 4th convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[3], name='d_conv3')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn3'))

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

  z_train_d = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])  
  g_model_train_d = generator(z_train_d, reuse=False)
  dg_model_train_d = discriminator(g_model_train_d, reuse=True)

  z_train_g = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])
  g_model_train_g = generator(z_train_g, reuse=True)
  dg_model_train_g = discriminator(g_model_train_g, reuse=True)

  # Optimizers
  t_vars = tf.trainable_variables()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model))
  d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model_train_d))
  tf.summary.scalar('d_loss', d_loss)

  # optimizer for model D training
  d_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
      d_loss, global_step=global_step, var_list=[v for v in t_vars if 'Model_D/' in v.name])

  # optimizer for model G training
  g_loss = -tf.reduce_mean(tf.log(dg_model_train_g))
  # g_loss = tf.reduce_mean(1. - tf.log(dg_model_train_g))
  tf.summary.scalar('g_loss', g_loss)
  g_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
      g_loss, var_list=[v for v in t_vars if 'Model_G/' in v.name])

  # Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # sess = tf.InteractiveSession()
  # tf.initialize_all_variables().run()

  # Savers
  # saver = tf.train.Saver(max_to_keep=20)
  # checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
  # if checkpoint and not FLAGS.debug:
  #     print('Restoring from', checkpoint)
  #     saver.restore(sess, checkpoint)
  # summary = tf.merge_all_summaries()
  # summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)


  # Training loop
  for step in range(2 if FLAGS.debug else int(1e6)):    
    # update model D for k times
    d_loss_val = 0
    for k in range(FLAGS.train_steps_model_D):
      z_batch_D = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)

      # data x random shuffle 
      arr = np.arange(dataset.shape[0])
      np.random.shuffle(arr)
      data_step = dataset[arr[:], :, :, :]
      x_batch_D = data_step[0 : FLAGS.batch_size]

      # Update discriminator
      _, d_loss_val_cur = sess.run([d_trainer, d_loss], feed_dict={x: x_batch_D, z_train_d: z_batch_D})
      d_loss_val += d_loss_val_cur

    d_loss_val /= FLAGS.train_steps_model_D

    # update model G for one time 
    z_batch_G = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
    # sess.run(g_trainer, feed_dict={z: z_batch_G})
    _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z_train_g: z_batch_G})


    # Log details
    print ('====> The {}th training step || Model G Loss: {:.8f} || Model D Loss: {:.8f}'.format(step, g_loss_val, d_loss_val))
    # summary_str = sess.run(summary, feed_dict={x: images, z: z_batch})
    # summary_writer.add_summary(summary_str, global_step.eval())

    # Early stopping
    if np.isnan(g_loss_val) or np.isnan(g_loss_val):
      print('Early stopping')
      break

    # if step % 100 == 0:
    #     # Save samples
    #     if FLAGS.sampledir:
    #         samples = 64
    #         z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
    #         images = sess.run(g_model, feed_dict={z: z2})
    #         images = np.reshape(images, [samples, 28, 28])
    #         images = (images + 1.) / 2.
    #         scipy.misc.imsave(FLAGS.sampledir + '/sample.png', merge(images, [int(math.sqrt(samples))] * 2))

    #     # save model
    #     if not FLAGS.debug:
    #         checkpoint_file = os.path.join(FLAGS.logdir, 'checkpoint')
    #         saver.save(sess, checkpoint_file, global_step=global_step)

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

  # if not tf.gfile.Exists(FLAGS.logdir):
  #     tf.gfile.MakeDirs(FLAGS.logdir)
  # if FLAGS.sampledir and not tf.gfile.Exists(FLAGS.sampledir):
  #     tf.gfile.MakeDirs(FLAGS.sampledir)
  # if FLAGS.sampledir:
  #     sample()
  #     return
  # dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
  # if FLAGS.classifier:
  #     gan_class(dataset)
  # elif FLAGS.kmeans:
  #     kmeans(dataset)
  # else:
  #     mnist_gan(dataset)


if __name__ == '__main__':
  # tf.app.run()
  main()