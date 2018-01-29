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


'''
    Generator model G using U-net structure
    - Input img: input image data (sketches)
    - Input options: options to store variables
    - Input reuse: represent whether reuse the generator
    - Input training: represent whether feed forward in training approach
'''
def generator_unet(image, options, reuse=False, name="generator_unet"):
  # define dropout rate different for train and test
  dropout_rate = 0.5 if options.is_training else 1.0

  with tf.variable_scope(name):
    # image is N x 256 x 256 x input_c_dim
    if reuse:
      tf.get_variable_scope().reuse_variables()
    else:
      assert tf.get_variable_scope().reuse is False

    '''
      Encoder section
    '''
    # image is (N x 256 x 256 x input_c_dim)
    e1 = instance_norm(conv2d_slim(image, options.gf_dim, name='g_e1_conv'))
    # e1 is (N x 128 x 128 x options.gf_dim)
    e2 = instance_norm(conv2d_slim(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
    # e2 is (N x 64 x 64 x options.gf_dim*2)
    e3 = instance_norm(conv2d_slim(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
    # e3 is (N x 32 x 32 x options.gf_dim*4)
    e4 = instance_norm(conv2d_slim(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
    # e4 is (N x 16 x 16 x options.gf_dim*8)
    e5 = instance_norm(conv2d_slim(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
    # e5 is (N x 8 x 8 x options.gf_dim*8)
    e6 = instance_norm(conv2d_slim(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
    # e6 is (N x 4 x 4 x options.gf_dim*8)
    e7 = instance_norm(conv2d_slim(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
    # e7 is (N x 2 x 2 x options.gf_dim*8)
    e8 = instance_norm(conv2d_slim(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
    # e8 is (N x 1 x 1 x options.gf_dim*8)


    '''
      Decoder section
    '''
    # define output size
    s = options.output_size
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    
    # d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8), [options.batch_size, s128, s128, options.gf_dim*8], name='g_d1', with_w=True)
    d1 = deconv2d_slim(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
    d1 = tf.nn.dropout(d1, dropout_rate)
    d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
    # d1 is (N x 2 x 2 x options.gf_dim*8*2)

    # d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1), [options.batch_size, s64, s64, options.gf_dim*8], name='g_d2', with_w=True)
    d2 = deconv2d_slim(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
    d2 = tf.nn.dropout(d2, dropout_rate)
    d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
    # d2 is (4 x 4 x options.gf_dim*8*2)

    # d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2), [options.batch_size, s32, s32, options.gf_dim*8], name='g_d3', with_w=True)
    d3 = deconv2d_slim(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
    d3 = tf.nn.dropout(d3, dropout_rate)
    d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
    # d3 is (8 x 8 x options.gf_dim*8*2)

    # d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3), [options.batch_size, s16, s16, options.gf_dim*8], name='g_d4', with_w=True)
    d4 = deconv2d_slim(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
    d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
    # d4 is (16 x 16 x options.gf_dim*8*2)

    # d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4), [options.batch_size, s8, s8, options.gf_dim*4], name='g_d5', with_w=True)
    d5 = deconv2d_slim(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
    d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
    # d5 is (32 x 32 x options.gf_dim*4*2)

    # d6, d6_w, d6_b = deconv2d(tf.nn.relu(d5), [options.batch_size, s4, s4, options.gf_dim*2], name='g_d6', with_w=True)
    d6 = deconv2d_slim(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
    d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
    # d6 is (64 x 64 x options.gf_dim*2*2)

    # d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6), [options.batch_size, s2, s2, options.gf_dim], name='g_d7', with_w=True)
    d7 = deconv2d_slim(tf.nn.relu(d6), options.gf_dim, name='g_d7')
    d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
    # d7 is (128 x 128 x options.gf_dim*1*2)

    # d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7), [options.batch_size, s, s, options.output_c_dim], name='g_d8', with_w=True)
    d8 = deconv2d_slim(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
    # d8 is (256 x 256 x output_c_dim)

    return tf.nn.tanh(d8)


'''
    Generator model G using Residual block
    - Input img: input image data (sketches)
    - Input options: options to store variables
    - Input nineBlock: use 9 ResBlock if true, otherwise use 6 ResBlock
    - Input reuse: represent whether reuse the generator
    - Input training: represent whether feed forward in training approach
'''
def generator_resnet(image, options, reuse=False, nineBlock=True, name="generator_resnet"):

  with tf.variable_scope(name):
    # image is 256 x 256 x input_c_dim
    if reuse:
      tf.get_variable_scope().reuse_variables()
    else:
      assert tf.get_variable_scope().reuse is False
    
    '''
      Justin Johnson's model from the paper
      The network with 9 blocks consists of: 
        c7s1-32, d64, d128, R128, R128, R128, R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
      The network with 9 blocks consists of: 
        c7s1-32, d64, d128, R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
    '''
    c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    # c7s1-32 layer, c1 is (N x 256 x 256 x 32)
    c1 = tf.nn.relu(instance_norm(conv2d_slim(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))

    # d64 layer, d1 is (N x 128 x 128 x 64)
    d1 = tf.nn.relu(instance_norm(conv2d_slim(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))

    # d128 layer, d2 is (N x 64 x 64 x 128)
    d2 = tf.nn.relu(instance_norm(conv2d_slim(d1, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
    
    '''
      Define G network with 6 resnet blocks
      ResNet keep the data shape (N x 64 x 64 x 128)
    '''
    r1 = residule_block(d2, options.gf_dim*4, name='g_r1')
    r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
    r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
    r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
    r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
    r6 = residule_block(r5, options.gf_dim*4, name='g_r6')

    r_out = r6
    
    # define G network with 9 resnet blocks
    if nineBlock:
      r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
      r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
      r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
      r_out = r9

    # define output shape
    s = options.output_size
    s2= int(s/2)
    
    # u64 layer, u1 is (N x 128 x 128 x 64)
    # u1 = deconv2d(r_out, [options.batch_size, s2, s2, options.gf_dim*2], 3, 3, 2, 2, name='g_d1_dc')
    u1 = deconv2d_slim(r_out, options.gf_dim*2, 3, 2, name='g_d1_dc')
    u1 = tf.nn.relu(instance_norm(u1, 'g_d1_bn'))

    # u32 layer, u1 is (N x 256 x 256 x 32)
    # u2 = deconv2d(u1, [options.batch_size, s, s, options.gf_dim], 3, 3, 2, 2, name='g_d2_dc')
    u2 = deconv2d_slim(u1, options.gf_dim, 3, 2, name='g_d2_dc')
    u2 = tf.nn.relu(instance_norm(u2, 'g_d2_bn'))

    # c7s1-3 layer, out is (N x 256 x 256 x output_c_dim)
    out = tf.pad(u2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    # pred = tf.nn.tanh(conv2d(out, options.output_c_dim, 7, 7, 1, 1, padding='VALID', name='g_pred_c'))
    pred = tf.nn.tanh(conv2d_slim(out, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

    return pred 



'''
  Discriminator model D (patchGAN)
  - Input img: image data concatanated between sketch data and photos(real or fake) 
  - Output logit: the scalar to represent the prob that net belongs to the real data
'''
def discriminator(image, options, reuse=False, name="discriminator"):

  with tf.variable_scope(name):
    # image is 256 x 256 x input_c_dim
    if reuse:
      tf.get_variable_scope().reuse_variables()
    else:
      assert tf.get_variable_scope().reuse is False

    '''
      Justin Johnson's discriminator model from the paper:
      c64-c128-c256-c512 (c represents: 4x4 Conv - InstanceNorm - 0.2 LeakyRelu)
    '''
    h0 = lrelu(conv2d_slim(image, options.df_dim, name='d_h0_conv'))
    # h0 is (128 x 128 x self.df_dim)
    h1 = lrelu(instance_norm(conv2d_slim(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
    # h1 is (64 x 64 x self.df_dim*2)
    h2 = lrelu(instance_norm(conv2d_slim(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
    # h2 is (32x 32 x self.df_dim*4)
    h3 = lrelu(instance_norm(conv2d_slim(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
    # h3 is (32 x 32 x self.df_dim*8)
    
    h4 = conv2d_slim(h3, 1, s=1, name='d_h3_pred')
    # h4 is (32 x 32 x 1)
    return h4
