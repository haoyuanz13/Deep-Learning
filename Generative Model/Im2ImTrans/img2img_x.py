# from __future__ import division
# from __future__ import print_function
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange
from glob import glob
import time

from functools import partial

from utils import *
from model import *
from dataLoader import *

'''
  The image to image translation class
  - including generator, discriminator, model construction, train and test operations.
'''
class img2img(object):
  def __init__(self, sess, train_sketches, train_photos, test_sketches, test_photos, 
               interval_plot, interval_save, curve_dir, sampler_dir, 
               im_h=256, im_w=256, im_c_x=3, im_c_y=3,
               batch_size=1, sample_size=1, max_iteration=10000, 
               G_f_dim=64, D_f_dim=64, L1_lambda=10):
    '''
    Args:
      sess: tensorflow session
      im_h, im_w: the size of images (height and width)
      im_c_x: the channel number of input sketches data
      im_c_y: the channel number of input photoes data
      batch_size: the number of training data of each batch
      sample_size: the number of sample data
      max_iteration: the total number of epoches
      interval_plot: the epoch interval for each training loss plot
      interval_save: the epoch interval for sample generation and curve saving
      curve_dir: the directory to store loss curve
      sampler_dir: the directory to store samples
      G_f_dim: the dimension of model G filters in 1st conv layer
      D_f_dim: the dimension of model D filters in 1st conv layer
      L1_lambda: the weight of L1 loss for G loss
      train(test)_sketches: the sketches dataset for training(test)
      train(test)_photos: the photos dataset for training(test)
    '''
    self.sess = sess
    self.im_h, self.im_w = im_h, im_w
    self.im_c_skt, self.im_c_pht = im_c_x, im_c_y
    self.batch_size, self.sample_size = batch_size, sample_size
    self.max_epoch = max_iteration

    self.intval_plot, self.intval_save = interval_plot, interval_save
    self.curve_dir, self.sampler_dir = curve_dir, sampler_dir

    self.g_dim1, self.d_dim1 = G_f_dim, D_f_dim
    self.L1_lambda = L1_lambda

    self.train_skt, self.train_pht = train_sketches, train_photos
    self.test_skt, self.test_pht = test_sketches, test_photos

    '''
      define batch_norm layers to deal with poor initialization
    '''
    # BN layers in generator
    self.BN_2_GE = batch_norm(name='bn_2_ge')
    self.BN_3_GE = batch_norm(name='bn_3_ge')
    self.BN_4_GE = batch_norm(name='bn_4_ge')
    self.BN_5_GE = batch_norm(name='bn_5_ge')
    self.BN_6_GE = batch_norm(name='bn_6_ge')
    self.BN_7_GE = batch_norm(name='bn_7_ge')
    self.BN_8_GE = batch_norm(name='bn_8_ge')

    self.BN_1_GD = batch_norm(name='bn_1_gd')
    self.BN_2_GD = batch_norm(name='bn_2_gd')
    self.BN_3_GD = batch_norm(name='bn_3_gd')
    self.BN_4_GD = batch_norm(name='bn_4_gd')
    self.BN_5_GD = batch_norm(name='bn_5_gd')
    self.BN_6_GD = batch_norm(name='bn_6_gd')
    self.BN_7_GD = batch_norm(name='bn_7_gd')

    # BN layers in discriminator
    self.BN_2_D = batch_norm(name='bn_2_d')
    self.BN_3_D = batch_norm(name='bn_3_d')
    self.BN_4_D = batch_norm(name='bn_4_d')
    self.BN_5_D = batch_norm(name='bn_5_d')
    self.BN_6_D = batch_norm(name='bn_6_d')

    # build model
    self.build_model()


  '''
    Generator model G
    - Input img: input image data (sketches)
    - Input reuse: represent whether reuse the generator
    - Input training: represent whether feed forward in training approach
  '''
  def generator(self, img, reuse=False):
    with tf.variable_scope('Model_G', reuse=reuse):
      '''
        Encoder section
      '''
      # 1st conv: Conv+lrelu || [N, im_h, im_w, 1]->[N, im_h/2, im_w/2, g_dim1]
      e1 = lrelu(conv2d(img, self.g_dim1, name='g_conv_1'))

      # 2nd conv: Conv+Bn+lrelu || [N, im_h/2, im_w/2, g_dim1]->[N, im_h/4, im_w/4, 2*g_dim1]
      e2 = lrelu(self.BN_2_GE(conv2d(e1, self.g_dim1*2, name='g_conv_2')))

      # 3rd conv: Conv+Bn+lrelu || [N, im_h/4, im_w/4, 2*g_dim1]->[N, im_h/8, im_w/8, 4*g_dim1]
      e3 = lrelu(self.BN_3_GE(conv2d(e2, self.g_dim1*4, name='g_conv_3')))

      # 4th conv: Conv+Bn+lrelu || [N, im_h/8, im_w/8, 4*g_dim1]->[N, im_h/16, im_w/16, 8*g_dim1]
      e4 = lrelu(self.BN_4_GE(conv2d(e3, self.g_dim1*8, name='g_conv_4')))

      # 5th conv: Conv+Bn+lrelu || [N, im_h/16, im_w/16, 8*g_dim1]->[N, im_h/32, im_w/32, 8*g_dim1]
      e5 = lrelu(self.BN_5_GE(conv2d(e4, self.g_dim1*8, name='g_conv_5')))
     
      # 6th conv: Conv+Bn+lrelu || [N, im_h/32, im_w/32, 8*g_dim1]->[N, im_h/64, im_w/64, 8*g_dim1]
      e6 = lrelu(self.BN_6_GE(conv2d(e5, self.g_dim1*8, name='g_conv_6')))
           
      # 7th conv: Conv+Bn+lrelu || [N, im_h/64, im_w/64, 8*g_dim1]->[N, im_h/128, im_w/128, 8*g_dim1]
      e7 = lrelu(self.BN_7_GE(conv2d(e6, self.g_dim1*8, name='g_conv_7')))

      # 8th conv: Conv+Bn+lrelu || [N, im_h/128, im_w/128, 8*g_dim1]->[N, im_h/256, im_w/256, 8*g_dim1]
      e8 = lrelu(self.BN_8_GE(conv2d(e7, self.g_dim1*8, name='g_conv_8')))


      '''
        Decoder section using U-Net structure
      '''
      # 1st deconv: Deconv+Bn+dropout+relu
      d1 = deconv2d(e8, [self.batch_size, 2, 2, self.g_dim1*8], name="g_deconv_1")
      d1 = tf.nn.dropout(self.BN_1_GD(d1), 0.5)
      d1 = tf.nn.relu(tf.concat([d1, e7], 3))  # skip connection d1-e7
      # d1 has shape [N, 2, 2, 2*8*g_dim1]

      # 2nd deconv: Deconv+Bn+dropout+relu
      d2 = deconv2d(d1, [self.batch_size, 4, 4, self.g_dim1*8], name="g_deconv_2")
      d2 = tf.nn.dropout(self.BN_2_GD(d2), 0.5)
      d2 = tf.nn.relu(tf.concat([d2, e6], 3))  # skip connection d2-e6
      # d2 has shape [N, 4, 4, 2*8*g_dim1]

      # 3rd deconv: Deconv+Bn+dropout+relu
      d3 = deconv2d(d2, [self.batch_size, 8, 8, self.g_dim1*8], name="g_deconv_3")
      d3 = tf.nn.dropout(self.BN_3_GD(d3), 0.5)
      d3 = tf.nn.relu(tf.concat([d3, e5], 3))  # skip connection d3-e5
      # d3 has shape [N, 8, 8, 2*8*g_dim1]

      # 4th deconv: Deconv+Bn+dropout+relu
      d4 = deconv2d(d3, [self.batch_size, 16, 16, self.g_dim1*8], name="g_deconv_4")
      d4 = tf.nn.dropout(self.BN_4_GD(d4), 0.5)
      d4 = tf.nn.relu(tf.concat([d4, e4], 3))  # skip connection d4-e4
      # d4 has shape [N, 16, 16, 2*8*g_dim1]

      # 5th deconv: Deconv+Bn+dropout+relu
      d5 = deconv2d(d4, [self.batch_size, 32, 32, self.g_dim1*4], name="g_deconv_5")
      d5 = tf.nn.dropout(self.BN_5_GD(d5), 0.5)
      d5 = tf.nn.relu(tf.concat([d5, e3], 3))  # skip connection d5-e3
      # d5 has shape [N, 32, 32, 2*4*g_dim1]

      # 6th deconv: Deconv+Bn+dropout+relu
      d6 = deconv2d(d5, [self.batch_size, 64, 64, self.g_dim1*2], name="g_deconv_6")
      d6 = tf.nn.dropout(self.BN_6_GD(d6), 0.5)
      d6 = tf.nn.relu(tf.concat([d6, e2], 3))  # skip connection d6-e2
      # d6 has shape [N, 64, 64, 2*2*g_dim1]

      # 7th deconv: Deconv+Bn+dropout+relu
      d7 = deconv2d(d6, [self.batch_size, 128, 128, self.g_dim1], name="g_deconv_7")
      d7 = tf.nn.dropout(self.BN_7_GD(d7), 0.5)
      d7 = tf.nn.relu(tf.concat([d7, e1], 3))  # skip connection d7-e1
      # d7 has shape [N, 128, 128, 2*g_dim1]

      # 8th deconv: Deconv+tanh
      d8 = tf.nn.tanh(deconv2d(d7, [self.batch_size, 256, 256, self.im_c_pht], name="g_deconv_8"))
      # d8 has shape [N, 256, 256, im_c_pht]

      return d8


  '''
    Discriminator model D
    - Input img: image data concatanated between sketch data and photos(real or fake) 
    - Output logit: the scalar to represent the prob that net belongs to the real data
  '''
  def discriminator(self, img, reuse=True):
    with tf.variable_scope('Model_D', reuse=reuse):
      # 1st conv+lrelu: [N, 256, 256, im_c_skt + im_c_pht]->[N, 128, 128, d_dim1] 
      y1 = lrelu(conv2d(img, self.d_dim1, name='d_conv_1'))

      # 2nd conv+bn+lrelu: [N, 128, 128, d_dim1]->[N, 64, 64, d_dim1*2]
      y2 = lrelu(self.BN_2_D(conv2d(y1, self.d_dim1*2, name='d_conv_2')))

      # 3rd conv+bn+lrelu: [N, 64, 64, d_dim1*2]->[N, 32, 32, d_dim1*4]
      y3 = lrelu(self.BN_3_D(conv2d(y2, self.d_dim1*4, name='d_conv_3')))

      # 4th conv+bn+lrelu: [N, 32, 32, d_dim1*4]->[N, 16, 16, d_dim1*8]
      y4 = lrelu(self.BN_4_D(conv2d(y3, self.d_dim1*8, name='d_conv_4')))

      logit = linear(tf.reshape(y4, [self.batch_size, -1]), 1, name='d_fc')

      # # 5th conv+bn+lrelu: [N, 16, 16, d_dim1*8]->[N, 8, 8, d_dim1*8]
      # y5 = lrelu(self.BN_5_D(conv2d(y4, self.d_dim1*8, name='d_conv_5')))

      # # 5th conv+bn+lrelu: [N, 8, 8, d_dim1*8]->[N, 4, 4, d_dim1*8]
      # y6 = lrelu(self.BN_6_D(conv2d(y5, self.d_dim1*8, name='d_conv_6')))

      # # fc: [N, 4*4*1] -> [N,1]
      # logit = linear(tf.reshape(y6, [self.batch_size, -1]), 1, name='d_fc')
      # logit = fc(y6, 1)

      return tf.nn.sigmoid(logit), logit


  '''
    load sample images randomly
  '''
  def load_random_samples(self, FLAGS):
    # random shuffle test dataset
    total_test = self.test_skt.shape[0]
    arr = np.arange(total_test)
    np.random.shuffle(arr)

    test_skt_sample = self.test_skt[arr[:], :, :, :]
    test_pht_sample = self.test_pht[arr[:], :, :, :]

    # test data for reconstruction
    x_ipt_sample = test_skt_sample[:self.batch_size, :, :, :]
    x_ipt_sample_input = addNoise(x_ipt_sample) if FLAGS.addNoise else x_ipt_sample

    x_ipt_sample_target = test_pht_sample[:self.batch_size, :, :, :]

    return x_ipt_sample_input, x_ipt_sample_target


  '''
    Model Construction
    - includes loss cpmputation
  '''
  def build_model(self):
    # tensor to store sketches and photos
    self.img_skt = tf.placeholder(tf.float32, shape=[self.batch_size, self.im_h, self.im_w, self.im_c_skt], name='sketch_img')
    self.img_pht = tf.placeholder(tf.float32, shape=[self.batch_size, self.im_h, self.im_w, self.im_c_pht], name='photo_img')

    # test data for sampler
    self.img_sampler = tf.placeholder(tf.float32, shape=[self.batch_size, self.im_h, self.im_w, self.im_c_skt], name='test_sketch_img')

    # generator to obtain fake_pht
    self.fake_pht = self.generator(self.img_skt, reuse=False)

    # discriminators (skt + real_pht or fake_pht)
    self.sktAndreal_pht = tf.concat([self.img_skt, self.img_pht], 3)
    self.sktAndfake_pht = tf.concat([self.img_skt, self.fake_pht], 3)

    self.prob_real_sig, self.prob_real_logits = self.discriminator(self.sktAndreal_pht, reuse=False)
    self.prob_fake_sig, self.prob_fake_logits = self.discriminator(self.sktAndfake_pht, reuse=True)

    # sample data
    self.fake_samplers_skt = self.generator(self.img_sampler, reuse=True)

    '''
      loss computation
    '''
    # D_loss = -log[D(skt,real_pht)] - log[1 - D(skt,fake_pht)]
    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=tf.ones_like(self.prob_real_sig), logits=self.prob_real_logits))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=tf.zeros_like(self.prob_fake_sig), logits=self.prob_fake_logits))

    self.d_loss = self.d_loss_real + self.d_loss_fake

    # G_loss = -log[D(skt,fake_pht)] + L1_loss(fake_pht, real_pht)
    self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=tf.ones_like(self.prob_fake_sig), logits=self.prob_fake_logits))
    self.g_loss_l1 = tf.reduce_mean(tf.abs(self.img_pht - self.fake_pht))

    self.g_loss = self.g_loss_adv + self.L1_lambda * self.g_loss_l1

    '''
      Obtain variables
    '''
    model_vars = tf.trainable_variables()
    self.model_D_vars = [v for v in model_vars if 'Model_D/' in v.name]
    self.model_G_vars = [v for v in model_vars if 'Model_G/' in v.name]

    # define saver
    self.saver = tf.train.Saver()

    print "\n>>>>>>>> The [NEW] img2img model has been well initialized! \n"
    return

  '''
    Main train code
  '''
  def train(self, FLAGS):
    # optimizer for model D
    d_trainer = tf.train.AdamOptimizer(FLAGS.lr_modelD, beta1=FLAGS.beta1).minimize(
        self.d_loss, var_list=self.model_D_vars)

    # optimizer for model G
    g_trainer = tf.train.AdamOptimizer(FLAGS.lr_modelG, beta1=FLAGS.beta1).minimize(
        self.g_loss, var_list=self.model_G_vars)

    # Session
    self.sess.run(tf.global_variables_initializer())

    d_loss_set,  g_loss_set = np.zeros([int(self.max_epoch)]), np.zeros([int(self.max_epoch)])

    # load train data file names
    data_skt_path = sorted(glob('./data/cufs_students/sketches_train/*.jpg'))
    data_pht_path = sorted(glob('./data/cufs_students/photos_train/*.jpg'))

    # Outer training loop (control epoch)
    for epoch in xrange(2 if FLAGS.debug else int(self.max_epoch)): 
      print '\n<===================== The {}th Epoch training is processing =====================>'.format(epoch)
      # data random shuffle 
      arr = np.arange(FLAGS.total_num)
      np.random.shuffle(arr)

      total_step = FLAGS.total_num / self.batch_size
      # obtain random sample data
      sample_skts_epoch, sample_phts_epoch = self.load_random_samples(FLAGS)

      # inner training loop (control step)
      train_d_loss, train_g_loss = 0, 0
      start_time = time.time()
      for step in xrange(0, total_step):
        skt_batch = imload(data_skt_path[arr[step]])
        pht_batch = imload(data_pht_path[arr[step]])

        skt_input = addNoise(skt_batch) if FLAGS.addNoise else skt_batch  # add noise if necessary

        feed_dict_train = {self.img_skt:skt_input, self.img_pht:pht_batch, self.img_sampler:sample_skts_epoch}
        
        # train model D for n_critic_D times first
        for t in xrange(0, FLAGS.n_critic_D):
         self.sess.run(d_trainer, feed_dict=feed_dict_train)

        # train model G for n_critic_G times later
        for t in xrange(0, FLAGS.n_critic_G):
          self.sess.run(g_trainer, feed_dict=feed_dict_train)

        # evaluate loss
        d_loss_batch = self.d_loss.eval(feed_dict=feed_dict_train)
        g_loss_batch = self.g_loss.eval(feed_dict=feed_dict_train)

        print ('Epoch: [{}] [{:02d}/{}] || Time: {:.4f}s || D Loss: {:.8f} || G Loss: {:.8f}'.format(
          epoch, step + 1, total_step, time.time() - start_time, d_loss_batch, g_loss_batch))

        # update loss sum
        train_d_loss += d_loss_batch
        train_g_loss += g_loss_batch

      # end inner loop

      # store training loss
      d_loss_set[epoch], g_loss_set[epoch] = train_d_loss/total_step, train_g_loss/total_step

      # Log details
      if epoch % self.intval_plot == 0:
        print '\n-------------------------------------------------------------------------------------------------------------------------------'
        print ('The {}th training epoch completed || Total time cost {:.4f}s || Model D Avg Loss {:.8f} || Model G Avg Loss: {:.8f}'.format(
          epoch, time.time() - start_time, train_d_loss/total_step, train_g_loss/total_step))

      # Early stopping
      if np.isnan(train_d_loss) or np.isnan(train_g_loss):
        print('Early stopping')
        break

      # plot generative images
      if epoch % self.intval_save == 0:
        print ('\n===========> Generate sample images and saving .................................')
        if self.sampler_dir:
          samples = self.sample_size

          # execute sampler
          images, im_d_loss, im_g_loss = \
                  self.sess.run([self.fake_samplers_skt, self.d_loss, self.g_loss], feed_dict=feed_dict_train)

          print("[Sample Loss] Model D Loss: {:.8f} || Model G Loss: {:.8f}".format(im_d_loss, im_g_loss))

          images = images[:samples, :, :, :]  # obtain certain number of images
          images = (images + 1.) / 2.  # image recover

          # concat images
          im_concat = np.zeros([self.im_h, 3 * self.im_w, self.im_c_pht])
          im_concat[:, :self.im_w, :] = sample_skts_epoch[0, :, :, :]
          im_concat[:, self.im_w:2*self.im_w, :] = images[0, :, :, :]
          im_concat[:, 2*self.im_w:, :] = sample_phts_epoch[0, :, :, :]

          scipy.misc.imsave(self.sampler_dir + ('/sample_{:04d}.png'.format(epoch)), im_concat)
          
          # save loss curve
          np.save(self.curve_dir + '/loss_d.npy', d_loss_set[:epoch + 1])
          np.save(self.curve_dir + '/loss_g.npy', g_loss_set[:epoch + 1])

          
          

