# from __future__ import division
# from __future__ import print_function
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange
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
  def __init__(self, sess, im_h=256, im_w=256, im_c_x=3, im_c_y=3,
               batch_size=1, sample_size=1, max_iteration=10000, G_f_dim=64, D_f_dim=64, L1_lambda=10):
    '''
    Args:
      sess: tensorflow session
      im_h, im_w: the size of images (height and width)
      im_c_x: the channel number of input sketches data
      im_c_y: the channel number of input photoes data
      batch_size: the number of training data of each batch
      sample_size: the number of sample data
      max_iteration: the total number of epoches
      G_f_dim: the dimension of model G filters in 1st conv layer
      D_f_dim: the dimension of model D filters in 1st conv layer
      L1_lambda: the weight of L1 loss for G loss
      train/test/val_sketches: the sketches dataset for training/test/validation
      train/test/val_photos: the photos dataset for training/test/validation
    '''
    self.sess = sess
    self.im_h, self.im_w = im_h, im_w
    self.im_c_skt, self.im_c_pht = im_c_x, im_c_y
    self.batch_size, self.sample_size = batch_size, sample_size
    self.max_epoch = max_iteration

    self.g_dim1, self.d_dim1 = G_f_dim, D_f_dim
    self.L1_lambda = L1_lambda
    
    # load data
    skt_dict, pht_dict = dataloader_cufs_students()
    self.train_skt, self.train_pht = skt_dict['train'], pht_dict['train']
    self.test_skt, self.test_pht = skt_dict['test'], pht_dict['test']
    self.val_skt, self.val_pht = skt_dict['val'], pht_dict['val']

    # build model
    self.build_model()


  '''
    Generator model G
    - Input img: input image data (sketches)
    - Input reuse: represent whether reuse the generator
    - Input training: represent whether feed forward in training approach
  '''
  def generator(self, img, reuse=False, training=True):
    # batch normalize layer
    batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
    bn = partial(batch_norm, is_training=training)

    '''
      The stacked Conv blocks for encoder: Conv + Bn + leaky_relu
    '''
    # standard convolutional layer using lrelu  
    conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
    lrelu = partial(leak_relu, leak=0.2)
    # stacked layer: conv + bn + leaky relu
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)


    '''
      The stacked Deconv block for decoder: Deconv + Bn (manually add dropout and relu in the forward section)
    '''
    # deconvolutional layer
    dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

    # stacked layer: dconv + bn + relu
    dconv_bn = partial(dconv, normalizer_fn=bn, activation_fn=None, biases_initializer=None)

    '''
      Forward operation of Model G
    '''
    with tf.variable_scope('Model_G', reuse=reuse):
      '''
        Encoder section
      '''
      # 1st conv: Conv+lrelu || [N, im_h, im_w, 1]->[N, im_h/2, im_w/2, g_dim1]
      e1 = lrelu(conv(img, self.g_dim1, 5, 2))

      # 2nd conv: Conv+Bn+lrelu || [N, im_h/2, im_w/2, g_dim1]->[N, im_h/4, im_w/4, 2*g_dim1]
      e2 = conv_bn_lrelu(e1, self.g_dim1*2, 5, 2)

      # 3rd conv: Conv+Bn+lrelu || [N, im_h/4, im_w/4, 2*g_dim1]->[N, im_h/8, im_w/8, 4*g_dim1]
      e3 = conv_bn_lrelu(e2, self.g_dim1*4, 5, 2)

      # 4th conv: Conv+Bn+lrelu || [N, im_h/8, im_w/8, 4*g_dim1]->[N, im_h/16, im_w/16, 8*g_dim1]
      e4 = conv_bn_lrelu(e3, self.g_dim1*8, 5, 2)

      # 5th conv: Conv+Bn+lrelu || [N, im_h/16, im_w/16, 8*g_dim1]->[N, im_h/32, im_w/32, 8*g_dim1]
      e5 = conv_bn_lrelu(e4, self.g_dim1*8, 5, 2)
     
      # 6th conv: Conv+Bn+lrelu || [N, im_h/32, im_w/32, 8*g_dim1]->[N, im_h/64, im_w/64, 8*g_dim1]
      e6 = conv_bn_lrelu(e5, self.g_dim1*8, 5, 2)
           
      # 7th conv: Conv+Bn+lrelu || [N, im_h/64, im_w/64, 8*g_dim1]->[N, im_h/128, im_w/128, 8*g_dim1]
      e7 = conv_bn_lrelu(e6, self.g_dim1*8, 5, 2)

      # 8th conv: Conv+Bn+lrelu || [N, im_h/128, im_w/128, 8*g_dim1]->[N, im_h/256, im_w/256, 8*g_dim1]
      e8 = conv_bn_lrelu(e7, self.g_dim1*8, 5, 2)  # e8 has size [N, 1, 1, 8*g_dim1]


      '''
        Decoder section using U-Net structure
      '''
      # 1st deconv: Deconv+Bn+dropout+relu
      d1 = tf.nn.dropout(dconv_bn(e8, self.g_dim1*8, 5, 2), 0.5)
      d1 = tf.nn.relu(tf.concat([d1, e7], 3))  # skip connection d1-e7
      # d1 has shape [N, 2, 2, 2*8*g_dim1]

      # 2nd deconv: Deconv+Bn+dropout+relu
      d2 = tf.nn.dropout(dconv_bn(d1, self.g_dim1*8, 5, 2), 0.5)
      d2 = tf.nn.relu(tf.concat([d2, e6], 3))  # skip connection d2-e6
      # d2 has shape [N, 4, 4, 2*8*g_dim1]

      # 3rd deconv: Deconv+Bn+dropout+relu
      d3 = tf.nn.dropout(dconv_bn(d2, self.g_dim1*8, 5, 2), 0.5)
      d3 = tf.nn.relu(tf.concat([d3, e5], 3))  # skip connection d3-e5
      # d3 has shape [N, 8, 8, 2*8*g_dim1]

      # 4th deconv: Deconv+Bn+dropout+relu
      d4 = dconv_bn(d3, self.g_dim1*8, 5, 2)
      d4 = tf.nn.relu(tf.concat([d4, e4], 3))  # skip connection d4-e4
      # d4 has shape [N, 16, 16, 2*8*g_dim1]

      # 5th deconv: Deconv+Bn+dropout+relu
      d5 = dconv_bn(d4, self.g_dim1*4, 5, 2)
      d5 = tf.nn.relu(tf.concat([d5, e3], 3))  # skip connection d5-e3
      # d5 has shape [N, 32, 32, 2*4*g_dim1]

      # 6th deconv: Deconv+Bn+dropout+relu
      d6 = dconv_bn(d5, self.g_dim1*2, 5, 2)
      d6 = tf.nn.relu(tf.concat([d6, e2], 3))  # skip connection d6-e2
      # d6 has shape [N, 64, 64, 2*2*g_dim1]

      # 7th deconv: Deconv+Bn+dropout+relu
      d7 = dconv_bn(d6, self.g_dim1, 5, 2)
      d7 = tf.nn.relu(tf.concat([d7, e1], 3))  # skip connection d7-e1
      # d7 has shape [N, 128, 128, 2*g_dim1]

      # 8th deconv: Deconv+tanh
      d8 = tf.nn.tanh(dconv(d7, self.im_c_pht, 5, 2))
      # d8 has shape [N, 256, 256, im_c_pht]

      return d8

  '''
    Discriminator model D
    - Input img: image data concatanated between sketch data and photos(real or fake) 
    - Output logit: the scalar to represent the prob that net belongs to the real data
  '''
  def discriminator(self, img, reuse=True, training=True):
    # batch normalize layer
    batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
    bn = partial(batch_norm, is_training=training)

    # standard convolutional layer using lrelu  
    conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
    lrelu = partial(leak_relu, leak=0.2)

    # stacked layer: conv + bn + leaky relu
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    # fully connected layer
    fc = partial(flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

    # model D
    with tf.variable_scope('Model_D', reuse=reuse):
      # 1st conv+lrelu: [N, 256, 256, im_c_skt + im_c_pht]->[N, 128, 128, d_dim1] 
      y1 = lrelu(conv(img, self.d_dim1, 5, 2))

      # 2nd conv+bn+lrelu: [N, 128, 128, d_dim1]->[N, 64, 64, d_dim1*2]
      y2 = conv_bn_lrelu(y1, self.d_dim1 * 2, 5, 2)

      # 3rd conv+bn+lrelu: [N, 64, 64, d_dim1*2]->[N, 32, 32, d_dim1*4]
      y3 = conv_bn_lrelu(y2, self.d_dim1 * 4, 5, 2)

      # 4th conv+bn+lrelu: [N, 32, 32, d_dim1*4]->[N, 16, 16, d_dim1*8]
      y4 = conv_bn_lrelu(y3, self.d_dim1 * 8, 5, 2)

      # 5th conv+bn+lrelu: [N, 16, 16, d_dim1*4]->[N, 8, 8, d_dim1*8]
      y5 = conv_bn_lrelu(y4, self.d_dim1 * 8, 5, 2)

      # 5th conv+bn+lrelu: [N, 8, 8, d_dim1*8]->[N, 4, 4, d_dim1*8]
      y6 = conv_bn_lrelu(y5, self.d_dim1 * 8, 5, 2)

      # fc: [N, 4*4*d_dim1*8] -> [N,1]
      logit = fc(y6, 1)

      return tf.nn.sigmoid(logit), logit


  '''
    Model Construction
    - includes loss cpmputation
  '''
  def build_model(self):
    # tensor to store sketches and photos
    self.img_skt = tf.placeholder(tf.float32, shape=[self.batch_size, self.im_h, self.im_w, self.im_c_skt], name='sketch_img')
    self.img_pht = tf.placeholder(tf.float32, shape=[self.batch_size, self.im_h, self.im_w, self.im_c_pht], name='photo_img')

    # generator to obtain fake_pht
    self.fake_pht = self.generator(self.img_skt, reuse=False, training=True)

    # discriminators (skt + real_pht or fake_pht)
    self.sktAndreal_pht = tf.concat([self.img_skt, self.img_pht], 3)
    self.sktAndfake_pht = tf.concat([self.img_skt, self.fake_pht], 3)

    self.prob_real_sig, self.prob_real_logits = self.discriminator(self.sktAndreal_pht, reuse=False)
    self.prob_fake_sig, self.prob_fake_logits = self.discriminator(self.sktAndfake_pht, reuse=True)

    # sample data
    self.fake_samplers_skt = self.generator(self.img_skt, reuse=True, training=True)

    # summaries
    self.d_real_sum = tf.summary.histogram("d_real_sum", self.prob_real_sig)
    self.d_fake_sum = tf.summary.histogram("d_fake_sum", self.prob_fake_sig)
    self.fake_B_sum = tf.summary.image("fake_B", self.fake_pht)

    '''
      loss computation
    '''
    # D_loss = -log[D(skt,real_pht)] - log[1 - D(skt,fake_pht)]
    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=tf.ones_like(self.prob_real_sig), logits=self.prob_real_logits))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                          labels=tf.zeros_like(self.prob_fake_sig), logits=self.prob_fake_logits))

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    # G_loss = -log[D(skt,fake_pht)] + L1_loss(fake_pht, real_pht)
    self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                      labels=tf.ones_like(self.prob_fake_sig), logits=self.prob_fake_logits))
    self.g_loss_l1 = tf.reduce_mean(tf.abs(self.img_pht - self.fake_pht))

    self.g_loss = self.g_loss_adv + self.L1_lambda * self.g_loss_l1
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

    '''
      Obtain variables
    '''
    model_vars = tf.trainable_variables()
    self.model_D_vars = [v for v in model_vars if 'Model_D/' in v.name]
    self.model_G_vars = [v for v in model_vars if 'Model_G/' in v.name]

    # define saver
    self.saver = tf.train.Saver()

    print "\n[*] The model has been initialized SUCCESS! \n"
    return

  '''
    Main train code
  '''
  def train(self, args):
    # optimizer for model D
    d_trainer = tf.train.AdamOptimizer(args.lr_modelD, beta1=args.beta1).minimize(
        self.d_loss, var_list=self.model_D_vars)

    # optimizer for model G
    g_trainer = tf.train.AdamOptimizer(args.lr_modelG, beta1=args.beta1).minimize(
        self.g_loss, var_list=self.model_G_vars)

    # Session
    self.sess.run(tf.global_variables_initializer())

    # add summaries
    self.g_sum = tf.summary.merge([self.d_fake_sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

    if self.load_checkpoint(args, args.checkpoint_dir):
      print("[*] Checkepoint Load SUCCESS")
    else:
      print("[!] Checkpoint Load failed...")

    d_loss_set,  g_loss_set = np.zeros([int(self.max_epoch)]), np.zeros([int(self.max_epoch)])
    counter = 1
    # Outer training loop (control epoch)
    for epoch in xrange(2 if args.debug else int(self.max_epoch)): 
      print '\n<===================== The {}th Epoch training is processing =====================>'.format(epoch)
      # data x random shuffle
      train_size_min = min(args.train_size, self.train_skt.shape[0])
      arr = np.arange(train_size_min)
      np.random.shuffle(arr)

      skt_epoch = self.train_skt[arr[:], :, :, :]
      pht_epoch = self.train_pht[arr[:], :, :, :]

      total_step = train_size_min / self.batch_size

      # inner training loop (control step)
      train_d_loss, train_g_loss = 0, 0
      start_time = time.time()
      for step in xrange(0, total_step):
        skt_batch = skt_epoch[step * self.batch_size : (step + 1) * self.batch_size]
        pht_batch = pht_epoch[step * self.batch_size : (step + 1) * self.batch_size]

        skt_input = addNoise(skt_batch) if args.addNoise else skt_batch  # add noise if necessary

        feed_dict_train = {self.img_skt:skt_input, self.img_pht:pht_batch}
        
        # train model D for n_critic_D times first
        for t in xrange(0, args.n_critic_D):
         _, summary_d = self.sess.run([d_trainer, self.d_sum], feed_dict=feed_dict_train)
         self.writer.add_summary(summary_d, counter)

        # train model G for n_critic_G times later
        for t in xrange(0, args.n_critic_G):
          _, summary_g = self.sess.run([g_trainer, self.g_sum], feed_dict=feed_dict_train)
          self.writer.add_summary(summary_g, counter)

        # evaluate loss
        d_loss_batch = self.d_loss.eval(feed_dict=feed_dict_train)
        g_loss_batch = self.g_loss.eval(feed_dict=feed_dict_train)

        # update loss sum
        train_d_loss += d_loss_batch
        train_g_loss += g_loss_batch

        counter += 1
        print ('Epoch: [{}] [{:02d}/{}] || Time: {:.4f}s || D Loss: {:.8f} || G Loss: {:.8f}'.format(
          epoch, step + 1, total_step, time.time() - start_time, d_loss_batch, g_loss_batch))

      # end inner loop

      # Early stopping if happens nan value
      if np.isnan(train_d_loss) or np.isnan(train_g_loss):
        print('Early stopping')
        break
      
      # store training loss
      d_loss_set[epoch], g_loss_set[epoch] = train_d_loss/total_step, train_g_loss/total_step

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
        self.save_checkpoint(args, args.checkpoint_dir, epoch)
      
      # plot Avg loss
      if epoch % args.interval_plot == 0:
        print ('\n-------------------------------------------------------------------------------------------------------------------------------')
        print ('The {}th training epoch completed || Total time cost {:.4f}s || Model D Avg Loss {:.8f} || Model G Avg Loss: {:.8f}'.format(
            epoch, time.time() - start_time, train_d_loss/float(total_step), train_g_loss/float(total_step)))


  '''
    load sample images from Val dataset randomly
  '''
  def load_random_samples(self, args, random_load=True):
    if random_load:
      # random shuffle test dataset
      total_val = self.val_skt.shape[0]
      arr = np.arange(total_val)
      np.random.shuffle(arr)

      val_skt_sample = self.val_skt[arr[:], :, :, :]
      val_pht_sample = self.val_pht[arr[:], :, :, :]

    else:
      val_skt_sample = self.val_skt
      val_pht_sample = self.val_pht

    # test data for reconstruction
    x_ipt_sample = val_skt_sample[:self.batch_size, :, :, :]
    x_ipt_sample_input = addNoise(x_ipt_sample) if args.addNoise else x_ipt_sample

    x_ipt_sample_target = val_pht_sample[:self.batch_size, :, :, :]

    return x_ipt_sample_input, x_ipt_sample_target


  '''
    generate sample data
  '''
  def sample_model(self, args, sample_dir, epoch):
    # obtain random sample data
    sample_skts, sample_phts = self.load_random_samples(args, random_load=args.randomSample)
    samples, d_loss, g_loss = self.sess.run(
        [self.fake_samplers_skt, self.d_loss, self.g_loss], 
        feed_dict={self.img_skt:sample_skts, self.img_pht:sample_phts})

    # save samples
    self.save_samples(sample_dir, epoch, sample_skts, samples, sample_phts, concat=args.concatSamples)
    print("======>> [Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


  '''
    save generated images
  '''
  def save_samples(self, sample_dir, epoch, skt, samples, pht, concat=True):
    skt = merge(skt, [self.batch_size, 1])
    samples = merge(inverse_transform(samples), [self.batch_size, 1])
    pht = merge(pht, [self.batch_size, 1])
    
    # concat three images into single one
    if concat:
      im_concat = np.zeros((self.im_h, 3 * self.im_w, self.im_c_pht)).astype(np.float32)
      im_concat[:, :self.im_w, :] = skt
      im_concat[:, self.im_w:2*self.im_w, :] = samples
      im_concat[:, 2*self.im_w:, :] = pht

      scipy.misc.imsave(sample_dir + ('/sample_{:04d}.png'.format(epoch)), im_concat) 

    # save images separately
    else:
      scipy.misc.imsave(sample_dir + ('/sample_{:04d}.png'.format(epoch)), samples) 
      scipy.misc.imsave(sample_dir + ('/sample_photo.png'), pht) 
      scipy.misc.imsave(sample_dir + ('/sample_sketch.png'), skt) 


  '''
    save checkpoint for tensorboard
  '''
  def save_checkpoint(self, args, checkpoint_dir, step):
    model_name = "img2img.model"
    model_dir = "%s_%s_%s" % (args.dataset_name, self.batch_size, self.im_h)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


  '''
    load checkpoint
  '''
  def load_checkpoint(self, args, checkpoint_dir):
    print("===>> Reading checkpoint .....")

    model_dir = "%s_%s_%s" % (args.dataset_name, self.batch_size, self.im_h)
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
    
    print 'Test image has shape: ', self.test_skt.shape, '\n'

    start_time = time.time()    
    if self.load_checkpoint(args, args.checkpoint_dir):
      print("[*] Load SUCCESS\n")
    else:
      print("[!] Load failed...\n")

    test_total = self.test_skt.shape[0]
    for i in xrange(0, test_total):
      print 'Testing the {}th image .....'.format(i + 1)
      test_skt_cur = self.test_skt[i:i+1, :, :, :]
      test_pht_cur = self.test_pht[i:i+1, :, :, :]

      samples = self.sess.run(self.fake_samplers_skt, 
        feed_dict={self.img_skt:test_skt_cur, self.img_pht:test_pht_cur})

      # save test samples
      self.save_samples(args.test_dir, i+1, test_skt_cur, samples, test_pht_cur, concat=args.concatSamples)

          
          

