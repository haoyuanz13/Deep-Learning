from __future__ import division
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


class img2img_x(object):
  def __init__(self, sess, im_height=256, im_width=256, batch_size=1, sample_size=1, output_size=256,
               gf_dim=64, df_dim=64, L1_lambda=100, input_c_dim=3, output_c_dim=3, dataset_name='cufs_std_concat',
               checkpoint_dir=None, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      output_size: (optional) The resolution in pixels of the images. [256]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
      output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_grayscale = (input_c_dim == 1)
    self.batch_size = batch_size
    self.im_h = im_height
    self.im_w = im_width
    self.sample_size = sample_size
    self.output_size = output_size

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.input_c_dim = input_c_dim
    self.output_c_dim = output_c_dim

    self.L1_lambda = L1_lambda

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')
    self.d_bn6 = batch_norm(name='d_bn6')
    self.d_bn7 = batch_norm(name='d_bn7')

    self.g_bn_e2 = batch_norm(name='g_bn_e2')
    self.g_bn_e3 = batch_norm(name='g_bn_e3')
    self.g_bn_e4 = batch_norm(name='g_bn_e4')
    self.g_bn_e5 = batch_norm(name='g_bn_e5')
    self.g_bn_e6 = batch_norm(name='g_bn_e6')
    self.g_bn_e7 = batch_norm(name='g_bn_e7')
    self.g_bn_e8 = batch_norm(name='g_bn_e8')

    self.g_bn_d1 = batch_norm(name='g_bn_d1')
    self.g_bn_d2 = batch_norm(name='g_bn_d2')
    self.g_bn_d3 = batch_norm(name='g_bn_d3')
    self.g_bn_d4 = batch_norm(name='g_bn_d4')
    self.g_bn_d5 = batch_norm(name='g_bn_d5')
    self.g_bn_d6 = batch_norm(name='g_bn_d6')
    self.g_bn_d7 = batch_norm(name='g_bn_d7')

    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.build_model()


  '''
    Generator model G
    - Input img: input image data (sketches)
    - Input reuse: represent whether reuse the generator
    - Input training: represent whether feed forward in training approach
  '''
  def generator(self, image, reuse=False):
    with tf.variable_scope("generator") as scope:

      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse == False


      '''
        Encoder section
      '''
      # image is (N x 256 x 256 x input_c_dim)
      e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
      # e1 is (N x 128 x 128 x self.gf_dim)
      e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
      # e2 is (N x 64 x 64 x self.gf_dim*2)
      e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
      # e3 is (N x 32 x 32 x self.gf_dim*4)
      e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
      # e4 is (N x 16 x 16 x self.gf_dim*8)
      e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
      # e5 is (N x 8 x 8 x self.gf_dim*8)
      e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
      # e6 is (N x 4 x 4 x self.gf_dim*8)
      e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
      # e7 is (N x 2 x 2 x self.gf_dim*8)
      e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
      # e8 is (N x 1 x 1 x self.gf_dim*8)


      '''
        Decoder section using U-Net structure
      '''
      # define output size
      s = self.output_size
      s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

      self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), 
          [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
      d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
      d1 = tf.concat([d1, e7], 3)
      # d1 is (N x 2 x 2 x self.gf_dim*8*2)

      self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
          [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
      d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
      d2 = tf.concat([d2, e6], 3)
      # d2 is (N x 4 x 4 x self.gf_dim*8*2)

      self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
          [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
      d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
      d3 = tf.concat([d3, e5], 3)
      # d3 is (N x 8 x 8 x self.gf_dim*8*2)

      self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
          [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
      d4 = self.g_bn_d4(self.d4)
      d4 = tf.concat([d4, e4], 3)
      # d4 is (N x 16 x 16 x self.gf_dim*8*2)

      self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
          [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
      d5 = self.g_bn_d5(self.d5)
      d5 = tf.concat([d5, e3], 3)
      # d5 is (N x 32 x 32 x self.gf_dim*4*2)

      self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
          [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
      d6 = self.g_bn_d6(self.d6)
      d6 = tf.concat([d6, e2], 3)
      # d6 is (N x 64 x 64 x self.gf_dim*2*2)

      self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
          [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
      d7 = self.g_bn_d7(self.d7)
      d7 = tf.concat([d7, e1], 3)
      # d7 is (N x 128 x 128 x self.gf_dim*1*2)

      self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
          [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
      # d8 is (N x 256 x 256 x output_c_dim)

      return tf.nn.tanh(self.d8)


  '''
    Discriminator model D
    - Input img: image data concatanated between sketch data and photos(real or fake) 
    - Output logit: the scalar to represent the prob that net belongs to the real data
  '''
  def discriminator(self, img, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      # image is N x 256 x 256 x (input_c_dim + output_c_dim)
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse == False
      
      # 1st conv+lrelu: [N, 256, 256, im_c_skt + im_c_pht]->[N, 128, 128, df_dim] 
      y1 = lrelu(conv2d(img, self.df_dim, name='d_conv_1'))

      # 2nd conv+bn+lrelu: [N, 128, 128, df_dim]->[N, 64, 64, df_dim*2]
      y2 = lrelu(self.d_bn1(conv2d(y1, self.df_dim*2, name='d_conv_2')))

      # 3rd conv+bn+lrelu: [N, 64, 64, df_dim*2]->[N, 32, 32, df_dim*4]
      y3 = lrelu(self.d_bn2(conv2d(y2, self.df_dim*4, name='d_conv_3')))

      # 4th conv+bn+lrelu: [N, 32, 32, df_dim*4]->[N, 16, 16, df_dim*8]
      y4 = lrelu(self.d_bn3(conv2d(y3, self.df_dim*8, name='d_conv_4')))

      # 5th conv+bn+lrelu: [N, 16, 16, df_dim*8]->[N, 8, 8, df_dim*8]
      y5 = lrelu(self.d_bn4(conv2d(y4, self.df_dim*8, name='d_conv_5')))

      # 6th conv+bn+lrelu: [N, 8, 8, df_dim*8]->[N, 4, 4, df_dim*8]
      y6 = lrelu(self.d_bn5(conv2d(y5, self.df_dim*8, name='d_conv_6')))

      # fc: [N, 4*4*1] -> [N,1]
      logit = linear(tf.reshape(y5, [self.batch_size, -1]), 1, name='d_fc')

    # logit = fc(y6, 1)

    return tf.nn.sigmoid(logit), logit


  '''
    Model construction
  '''
  def build_model(self):
    self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, 
                                                 self.input_c_dim + self.output_c_dim], name='real_A_and_B_images')

    # real_B represents the input photo(y)
    self.real_B = self.real_data[:, :, :, :self.input_c_dim]

    # real_A represents the input sketch(x)
    self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

    # generate the fake photo (y)
    self.fake_B = self.generator(self.real_A)

    self.real_AB = tf.concat([self.real_A, self.real_B], 3)
    self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

    self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
    self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

    # generate the fake samples
    self.fake_B_sample = self.generator(self.real_A, reuse=True)

    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

    # d loss computation: D_loss = -log[D(skt,real_pht)] - log[1 - D(skt,fake_pht)]
    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        
    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    # g loss computation: G_loss = -log[D(skt,fake_pht)] + L1_loss(fake_pht, real_pht)
    self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
    self.g_loss_l1 = tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
    
    self.g_loss = self.g_loss_adv + self.L1_lambda * self.g_loss_l1
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

    print "\n[*] The model has been initialized SUCCESS! \n"


  '''
    Main train code
  '''
  def train(self, args):
    """Train img2img"""
    d_optim = tf.train.AdamOptimizer(args.lr_modelD, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(args.lr_modelG, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    # add summaries
    self.g_sum = tf.summary.merge([self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)


    if self.load_checkpoint(self.checkpoint_dir):
      print("[*] Checkepoint Load SUCCESS")
    else:
      print("[!] Checkpoint Load failed...")

    d_loss_set,  g_loss_set = np.zeros([int(args.max_iteration)]), np.zeros([int(args.max_iteration)])
    counter = 1  # count total training step [max_iteration x steps]
    
    # training loop
    for epoch in xrange(2 if args.debug else int(args.max_iteration)):
      print '\n<===================== The {}th Epoch training is processing =====================>'.format(epoch)
      data = glob('./data/{}/train/*.jpg'.format(self.dataset_name))
      np.random.shuffle(data)
      batch_idxs = min(len(data), args.train_size) // self.batch_size

      train_d_loss, train_g_loss = 0, 0
      start_time = time.time()
      for idx in xrange(0, batch_idxs):
        batch_files = data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [load_data(batch_file) for batch_file in batch_files]
        
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        # train model D for n_critic_D times first
        for t in xrange(0, args.n_critic_D):
          _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.real_data: batch_images })
          self.writer.add_summary(summary_str, counter)

        # train model G for n_critic_G times later
        for t in xrange(0, args.n_critic_G):
          _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.real_data: batch_images })
          self.writer.add_summary(summary_str, counter)


        errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
        errD_real = self.d_loss_real.eval({self.real_data: batch_images})
        errG = self.g_loss.eval({self.real_data: batch_images})

        train_d_loss += errD_fake+errD_real
        train_g_loss += errG

        counter += 1
        print ('Epoch: [{}] [{:02d}/{}] || Time: {:.4f}s || D Loss: {:.8f} || G Loss: {:.8f}'.format(
          epoch, idx + 1, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

      # Early stopping if happens nan value
      if np.isnan(train_d_loss) or np.isnan(train_g_loss):
        print('Early stopping')
        break

      # save single epoch result
      d_loss_set[epoch], g_loss_set[epoch] = train_d_loss/float(batch_idxs), train_g_loss/float(batch_idxs) 

      # sample and save model
      if epoch % args.interval_sample == 0:
        print ('\nGenerate fake samples ..............')
        self.sample_model(args, args.sample_dir, epoch)
        # save loss curve
        np.save(args.curve_dir + '/loss_d.npy', d_loss_set[:epoch + 1])
        np.save(args.curve_dir + '/loss_g.npy', g_loss_set[:epoch + 1])

      # save model
      if epoch % args.interval_save == 0:
        print ('\nSaving models ......................')
        self.save_checkpoint(args.checkpoint_dir, epoch)
      
      # plot Avg loss
      if epoch % args.interval_plot == 0:
        print ('\n-------------------------------------------------------------------------------------------------------------------------------')
        print ('The {}th training epoch completed || Total time cost {:.4f}s || Model D Avg Loss {:.8f} || Model G Avg Loss: {:.8f}'.format(
            epoch, time.time() - start_time, train_d_loss/float(batch_idxs), train_g_loss/float(batch_idxs)))
      


  '''
    Load random samples from val dataset
  '''
  def load_random_samples(self, random_load=True):
    if random_load:
      data = np.random.choice(glob('./data/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
      sample = [load_data(sample_file) for sample_file in data]

    else:
      data = sorted(glob('./data/{}/val/*.jpg'.format(self.dataset_name)))
      sample = load_data(data[5], flip=False)
      sample = np.expand_dims(sample, axis=0)  # shape [1, height, width, 6]


    if (self.is_grayscale):
      sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_images = np.array(sample).astype(np.float32)

    return sample_images

  '''
    generate sample data
  '''
  def sample_model(self, args, sample_dir, epoch):
    sample_images = self.load_random_samples(random_load=args.randomSample)
    samples, d_loss, g_loss = self.sess.run(
        [self.fake_B_sample, self.d_loss, self.g_loss], feed_dict={self.real_data: sample_images})
    
    samples = inverse_transform(samples)
    pht = sample_images[:, :, :, :self.input_c_dim]
    skt = sample_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

    # save samples
    self.save_samples(sample_dir, epoch, skt, samples, pht, concat=args.concatSamples)
    print("======>> [Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


  '''
    save generated images
  '''
  def save_samples(self, sample_dir, epoch, skt, samples, pht, concat=True):
    skt = merge(skt, [self.batch_size, 1])
    samples = merge(samples, [self.batch_size, 1])
    pht = merge(pht, [self.batch_size, 1])
    
    # concat three images into single one
    if concat:
      im_concat = np.zeros((self.output_size, 3 * self.output_size, self.output_c_dim)).astype(np.float32)
      im_concat[:, :self.output_size, :] = skt
      im_concat[:, self.output_size:2*self.output_size, :] = samples
      im_concat[:, 2*self.output_size:, :] = pht

      scipy.misc.imsave(sample_dir + ('/sample_{:04d}.png'.format(epoch)), im_concat) 

    # save images separately
    else:
      scipy.misc.imsave(sample_dir + ('/sample_{:04d}.png'.format(epoch)), samples) 
      scipy.misc.imsave(sample_dir + ('/sample_photo.png'), pht) 
      scipy.misc.imsave(sample_dir + ('/sample_sketch.png'), skt) 


  '''
    save checkpoint for tensorboard
  '''
  def save_checkpoint(self, checkpoint_dir, step):
    model_name = "img2img.model"
    model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


  '''
    load checkpoint
  '''
  def load_checkpoint(self, checkpoint_dir):
    print("===>> Reading checkpoint .....")

    model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      return True

    else:
      return False


  '''
    test model
  '''
  def test(self, args):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    sample_files = glob('./data/{}/test/*.jpg'.format(self.dataset_name))

    # sort testing input
    n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
    sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

    # load testing input
    print("====>> Loading testing images ...")
    sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

    if (self.is_grayscale):
      sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_images = np.array(sample).astype(np.float32)


    sample_images = [sample_images[i:i+self.batch_size]
                     for i in xrange(0, len(sample_images), self.batch_size)]
    
    sample_images = np.array(sample_images)
    print 'Test image has shape: ', sample_images.shape, '\n'

    start_time = time.time()    
    if self.load_checkpoint(self.checkpoint_dir):
      print("[*] Load SUCCESS\n")
    else:
      print("[!] Load failed...\n")

    for i, sample_image in enumerate(sample_images):
      idx = i + 1
      print 'Testing the {}th image .....'.format(idx)
      samples = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: sample_image})

      samples = inverse_transform(samples)
      pht = sample_image[:, :, :, :self.input_c_dim]
      skt = sample_image[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

      # save test samples
      self.save_samples(args.test_dir, idx, skt, samples, pht, concat=args.concatSamples)


          
          

