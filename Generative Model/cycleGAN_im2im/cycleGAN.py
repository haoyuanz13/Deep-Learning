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
from collections import namedtuple

from functools import partial

from utils import *
from model import *
from modules import * 
from dataLoader import *


class img2img(object):
  def __init__(self, sess, phase, resG, logLoss, im_height=256, im_width=256, batch_size=1, sample_size=1, output_size=256,
               imgPool_size=50, gf_dim=64, df_dim=64, L1_lambda=100, input_c_dim=3, output_c_dim=3, dataset_name='cufs_std_concat',
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
    self.imgPool_size = imgPool_size


    self.discriminator = discriminator
    if resG:
      self.generator = generator_resnet
    else:
      self.generator = generator_unet

    if logLoss:
      self.obtainLoss = sce_criterion
    else:  # use least-square loss
      self.obtainLoss = mae_criterion

    # define options
    OPTIONS = namedtuple('OPTIONS', 'batch_size output_size gf_dim df_dim output_c_dim is_training')
    self.options = OPTIONS._make((batch_size, output_size, gf_dim, df_dim, output_c_dim, phase == 'train'))

    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.build_model()


  '''
    Model construction
  '''
  def build_model(self):
    self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, 
                                                 self.input_c_dim + self.output_c_dim], name='real_A_and_B_images')

    # real_A represents the input collection X
    self.real_A = self.real_data[:, :, :, :self.input_c_dim]

    # real_B represents the input collection Y
    self.real_B = self.real_data[:, :, :, self.input_c_dim : self.input_c_dim + self.output_c_dim]


    '''
      Model G loss = L1 * cycle_loss + adv_loss_A2B + adv_loss_B2A
    '''
    # 1st cycle: A -> G -> fake_B -> F -> fake_A 
    ## Genertor G: generate the fake photo (y) 
    self.fake_B_fromA = self.generator(self.real_A, self.options, reuse=False, name="generator_G_A2B")

    ## Generator F: generate the fake sketch (x)
    self.fake_A_fromA = self.generator(self.fake_B_fromA, self.options, reuse=False, name="generator_F_B2A")

    self.D_B_fake = self.discriminator(self.fake_B_fromA, self.options, reuse=False, name="discriminator_B")

  
    # 2nd cycle: B -> F -> fake_A -> G -> fake_B 
    ## Generator F
    self.fake_A_fromB = self.generator(self.real_B, self.options, reuse=True, name="generator_F_B2A")

    ## Generator G
    self.fake_B_fromB = self.generator(self.fake_A_fromB, self.options, reuse=True, name="generator_G_A2B") 

    self.D_A_fake = self.discriminator(self.fake_A_fromB, self.options, reuse=False, name="discriminator_A")

    ## cycle loss = |F(G(A)) - A| + |G(F(B)) - B|
    self.loss_cycle = abs_criterion(self.real_A, self.fake_A_fromA) + abs_criterion(self.real_B, self.fake_B_fromB)

    ## adversarial loss (A to fake_B) for model G
    self.adverloss_g_A2B = self.obtainLoss(self.D_B_fake, tf.ones_like(self.D_B_fake))

    ## adversarial loss (B to fake_A) for model G
    self.adverloss_g_B2A = self.obtainLoss(self.D_A_fake, tf.ones_like(self.D_A_fake))

    ## combine g loss
    self.g_loss = self.adverloss_g_A2B + self.adverloss_g_B2A + self.L1_lambda * self.loss_cycle


    # fake A and B samples obtained from image pool which stores a history of generated samples
    self.fake_A_sample = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, self.input_c_dim], name='fake_A_sample')
    self.fake_B_sample = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, self.output_c_dim], name='fake_B_sample')

    '''
      Model D loss = {log[D_B(B)] + log[1-D_B(G(A))]} + {log[D_A(A)] + log[1-D_A(F(B))]}
      - G(A) and F(B) are obtained from image pool randomly, respectively 
      - we can replace log loss with MSE loss
    '''    
    # d loss from A to B
    ## D_B(B)
    self.D_B_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminator_B")

    ## D_B(G(A))
    self.D_B_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminator_B")

    self.db_loss_real = self.obtainLoss(self.D_B_real, tf.ones_like(self.D_B_real))
    self.db_loss_fake = self.obtainLoss(self.D_B_fake_sample, tf.zeros_like(self.D_B_fake_sample))
    self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

    # d loss from B to A
    ## D_A(A)
    self.D_A_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminator_A")

    ## D_A(F(B))
    self.D_A_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminator_A")

    self.da_loss_real = self.obtainLoss(self.D_A_real, tf.ones_like(self.D_A_real))
    self.da_loss_fake = self.obtainLoss(self.D_A_fake_sample, tf.zeros_like(self.D_A_fake_sample))
    self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

    # combine d loss
    self.d_loss = self.da_loss + self.db_loss

    '''
      write summaries
    '''
    # Model G loss
    self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.adverloss_g_A2B + self.L1_lambda * self.loss_cycle)
    self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.adverloss_g_B2A + self.L1_lambda * self.loss_cycle)
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

    # merge model G summaries
    self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])

    # Model D loss
    self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
    self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
    self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
    self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
    self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)

    # merge model D summaries
    self.d_sum = tf.summary.merge(
        [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
         self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
         self.d_loss_sum]
    )

    '''
      Generate samplers or test
    '''
    self.val_A = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, self.input_c_dim], name='val_A')
    self.val_B = tf.placeholder(tf.float32, [self.batch_size, self.im_h, self.im_w, self.output_c_dim], name='val_B')

    self.valB = self.generator(self.val_A, self.options, reuse=True, name="generator_G_A2B")
    self.valA = self.generator(self.val_B, self.options, reuse=True, name="generator_F_B2A")


    '''
      obtain variables
    '''
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    self.g_vars = [var for var in t_vars if 'generator' in var.name]
    # for var in t_vars: print(var.name)
    
    '''
      Build model saver
    '''
    self.saver = tf.train.Saver()

    '''
      Build an image pool
    '''
    self.imgPool = ImagePool(self.imgPool_size)

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
      # data are stored in A and B folder separately
      # dataA and dataB should be obatined randomly
      dataA = glob('./data/{}/trainA/*.jpg'.format(self.dataset_name))
      dataB = glob('./data/{}/trainB/*.jpg'.format(self.dataset_name))

      num_dataA, num_dataB = len(dataA), len(dataB)

      # random shuffle dataA and B to remove possible paired ones
      np.random.shuffle(dataA)
      np.random.shuffle(dataB)

      batch_idxs = min(min(num_dataA, num_dataB), args.train_size) // self.batch_size

      train_d_loss, train_g_loss = 0, 0
      start_time = time.time()
      for idx in xrange(0, batch_idxs):
        # cnoncatenate two files name
        batch_files_trainA = dataA[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_files_trainB = dataB[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_files = list(zip(batch_files_trainA, batch_files_trainB))

        batch = [load_data(batch_file) for batch_file in batch_files]
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)
        # batch_images has shape (batch_size, im_h, im_w, input_c_dim + output_c_dim)

        
        '''
          update model: G first and then D
        '''
        # train model G for n_critic_G times later
        ## generate fake images A and B, wirte summary
        errG = 0
        for t in xrange(0, args.n_critic_G):
          _, summary_str, fake_A_cur, fake_B_cur, errG_cur = \
          self.sess.run([g_optim, self.g_sum, self.fake_A_fromB, self.fake_B_fromA, self.g_loss], feed_dict={ self.real_data: batch_images })

          errG += errG_cur
          self.writer.add_summary(summary_str, counter)

          # store fake iamges into the image pool
          [fake_A_picked, fake_B_picked] = self.imgPool([fake_A_cur, fake_B_cur])


        # train model D for n_critic_D times using the random picked fake A and B
        errD = 0
        for t in xrange(0, args.n_critic_D):
          _, summary_str, errD_cur = self.sess.run([d_optim, self.d_sum, self.d_loss], feed_dict={ 
            self.real_data: batch_images, self.fake_A_sample:fake_A_picked, self.fake_B_sample:fake_B_picked })
          
          errD += errD_cur
          self.writer.add_summary(summary_str, counter)


        train_d_loss += errD / args.n_critic_D
        train_g_loss += errG / args.n_critic_G

        counter += 1
        print ('Epoch: [{}] [{:02d}/{}] || Time: {:.4f}s || D Loss: {:.8f} || G Loss: {:.8f}'.format(
          epoch, idx + 1, batch_idxs, time.time() - start_time, errD / args.n_critic_D, errG / args.n_critic_G))

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
      dataA = np.random.choice(glob('./data/{}/valA/*.jpg'.format(self.dataset_name)), self.batch_size)
      dataB = np.random.choice(glob('./data/{}/valB/*.jpg'.format(self.dataset_name)), self.batch_size)

      sample_files = list(zip(dataA, dataB))
      sample = [load_data(sample_file, flip=False, is_test=True) for sample_file in sample_files]

    else:
      dataA = sorted(glob('./data/{}/valA/*.jpg'.format(self.dataset_name)))
      dataB = sorted(glob('./data/{}/valB/*.jpg'.format(self.dataset_name)))

      data = (dataA[-1], dataB[-1])
      sample = load_data(data, flip=False, is_test=True)
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
    
    # samples generation
    fake_A2B, fake_A2B2A, fake_B2A, fake_B2A2B = self.sess.run(
        [self.fake_B_fromA, self.fake_A_fromA, self.fake_A_fromB, self.fake_B_fromB], 
          feed_dict={self.real_data:sample_images})

    # sample loss
    d_loss, g_loss = self.sess.run(
        [self.d_loss, self.g_loss], 
          feed_dict={self.real_data:sample_images, self.fake_A_sample:fake_B2A, self.fake_B_sample: fake_A2B})

    sample_A = sample_images[:, :, :, :self.input_c_dim]
    sample_B = sample_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

    # save samples
    self.save_samples(sample_dir, epoch, sample_A, fake_A2B, fake_A2B2A, ABA=True, concat=args.concatSamples)
    self.save_samples(sample_dir, epoch, sample_B, fake_B2A, fake_B2A2B, ABA=False, concat=args.concatSamples)

    print("======>> [Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


  '''
    save generated images
  '''
  def save_samples(self, sample_dir, epoch, ori, ori_dot, ori_dot_dot, ABA=True, concat=True):
    # inverse images
    ori, ori_dot, ori_dot_dot = inverse_transform(ori), inverse_transform(ori_dot), inverse_transform(ori_dot_dot)
    
    # merge images
    ori = merge(ori, [self.batch_size, 1])
    ori_dot = merge(ori_dot, [self.batch_size, 1])
    ori_dot_dot = merge(ori_dot_dot, [self.batch_size, 1])
    
    # concat three images into single one
    if concat:
      im_concat = np.zeros((self.output_size, 3 * self.output_size, self.output_c_dim)).astype(np.float32)
      im_concat[:, :self.output_size, :] = ori
      im_concat[:, self.output_size:2*self.output_size, :] = ori_dot
      im_concat[:, 2*self.output_size:, :] = ori_dot_dot

      if ABA:
        scipy.misc.imsave(sample_dir + ('/sample_ABA{:04d}.png'.format(epoch)), im_concat) 
      else:
        scipy.misc.imsave(sample_dir + ('/sample_BAB{:04d}.png'.format(epoch)), im_concat)

    # save images separately
    else:
      if ABA:
        scipy.misc.imsave(sample_dir + ('/real_A.png'), ori)
        scipy.misc.imsave(sample_dir + ('/sample_ABA_fakeB{:04d}.png'.format(epoch)), ori_dot) 
        scipy.misc.imsave(sample_dir + ('/sample_ABA_fakeA{:04d}.png'.format(epoch)), ori_dot_dot) 

      else:
        scipy.misc.imsave(sample_dir + ('/real_B.png'), ori)
        scipy.misc.imsave(sample_dir + ('/sample_BAB_fakeA{:04d}.png'.format(epoch)), ori_dot) 
        scipy.misc.imsave(sample_dir + ('/sample_BAB_fakeB{:04d}.png'.format(epoch)), ori_dot_dot) 
         


  '''
    save checkpoint for tensorboard
  '''
  def save_checkpoint(self, checkpoint_dir, step):
    model_name = "cycleGAN_img2img.model"
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
    save test images
  '''
  def save_tests(self, test_dir, epoch, ori, target, AB=True, concat=True):
    # inverse images
    ori, target = inverse_transform(ori), inverse_transform(target)
    
    # merge images
    ori = merge(ori, [self.batch_size, 1])
    target = merge(target, [self.batch_size, 1])
    
    # concat three images into single one
    if concat:
      im_concat = np.zeros((self.output_size, 2 * self.output_size, self.output_c_dim)).astype(np.float32)
      im_concat[:, :self.output_size, :] = ori
      im_concat[:, self.output_size:, :] = target

      if AB:
        scipy.misc.imsave(test_dir + ('/test_AB{:04d}.png'.format(epoch)), im_concat) 
      else:
        scipy.misc.imsave(test_dir + ('/test_BA{:04d}.png'.format(epoch)), im_concat)

    # save images separately
    else:
      if AB:
        scipy.misc.imsave(test_dir + ('/real_A{:04d}.png'.format(epoch)), ori)
        scipy.misc.imsave(test_dir + ('/test_AB_fakeB{:04d}.png'.format(epoch)), target) 

      else:
        scipy.misc.imsave(test_dir + ('/real_B{:04d}.png'.format(epoch)), ori)
        scipy.misc.imsave(test_dir + ('/test_BA_fakeA{:04d}.png'.format(epoch)), target) 

  
  '''
    test model
  '''
  def test(self, args):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    sample_filesA = sorted(glob('./data/{}/testA/*.jpg'.format(self.dataset_name)))
    sample_filesB = sorted(glob('./data/{}/testB/*.jpg'.format(self.dataset_name)))

    # load testing input
    print("====>> Loading testing images ...")
    test_files = list(zip(sample_filesA, sample_filesB))
    sample = [load_data(test_file, flip=False, is_test=True) for test_file in test_files]

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
      print 'Testing the {}th instance .....'.format(idx)
      
      # test B->A cycle
      real_B = sample_image[:,:,:, self.input_c_dim : self.input_c_dim + self.output_c_dim]
      test_A = self.sess.run(self.valA, feed_dict={self.val_B: real_B})

      # test A->B cycle
      real_A = sample_image[:,:,:, :self.input_c_dim]
      test_B = self.sess.run(self.valB, feed_dict={self.val_A: real_A})

      # save test samples
      self.save_tests(args.test_dir, idx, real_A, test_B, AB=True, concat=args.concatSamples)
      self.save_tests(args.test_dir, idx, real_B, test_A, AB=False, concat=args.concatSamples)


          
          

