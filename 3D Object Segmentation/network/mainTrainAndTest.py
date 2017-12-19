'''
  File name: mainTrainAndTest.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains the main training and testing function
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image
import time

from layers import *
from utils_model import *
from models import *
from dataloader import *


################
# Define flags #
################
flags = tf.app.flags
flags.DEFINE_string("dir_train", "dataset/train/", "Directory to save training data")
flags.DEFINE_string("dri_test", "dataset/test/", "Directory to save test data")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("step", 100, "The step in each epoch iteration [100]")
flags.DEFINE_integer("iteration", 2000, "The max iteration times [1000]")
flags.DEFINE_integer("display_interval", 4, "The step interval to plot training loss and accuracy [10]")
flags.DEFINE_integer("test_interval", 50, "The step interval to test model [40]")
flags.DEFINE_integer("ind_base", 2, "The Conv block ind [0]")
flags.DEFINE_integer("n_class", 10, "The total number of class type [10]")
flags.DEFINE_float("weight_decay_ratio", 0.05, "weight decay ration [0.05]")
flags.DEFINE_float("threshold_reg", 0.005, "the threshold of correct bbox estimation [0.005]")
flags.DEFINE_float("threshold_depth", 0.005, "the threshold of correct depth estimation [0.005]")
FLAGS = flags.FLAGS


net_label = ['ConvNet', 'MobileNet', 'ResNet']

'''
  MVCNN 
  - Input views: NxVxHxWxC (N: batch size, V: number of views)
  - Input label: Nx1
'''
def MVCNN(views, label, bbox, depth_gt, bs, reuse, is_train, ind_base):
  with tf.variable_scope('MVCNN', reuse=reuse) as mvcnn:
    # obtain the number of views 
    num_views = views.get_shape().as_list()[1]
    input_channel = views.get_shape().as_list()[-1]

    # multiple views concat
    with tf.variable_scope('multiView_concat', reuse=reuse) as mv:
      # transpose views: [NxVxHxWxC] -> [VxNxHxWxC]
      views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
      view_list = []
      
      for i in xrange(num_views):
        # set reuse True for i > 0 in order to share weight and bias
        reuse_base = (i != 0)
        raw_v = tf.gather(views, i)  # obtain raw view with shape [NxHxWxC]
        var_base, feaMap_i = BaseNet(raw_v, input_channel, FLAGS.weight_decay_ratio, ind_base, reuse_base, is_train, name='CNN1')
        view_list.append(feaMap_i)  # feaMap_i has shape [batch size, 4, 4, 256]


    # cls branch
    with tf.variable_scope('cls', reuse=reuse) as cls:
      feav_list = feaMapFlatten(view_list)
      feaVec_cls = viewPooling(feav_list, 'viewPooling')  # the shape is Nxk

      # fully connected layer: convert from 4x4x128 to 1024
      with tf.variable_scope('FC1', reuse=reuse) as fc1:
        feaVec_cls = FcBlock(feaVec_cls, 4*4*256, 1024, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

      # fully connected layer: convert from 1024 to 256
      with tf.variable_scope('FC2', reuse=reuse) as fc2:
        feaVec_cls = FcBlock(feaVec_cls, 1024, 256, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

      # fully connected layer: convert from 256 to 10
      with tf.variable_scope('FC3', reuse=reuse) as fc3:
        feaVec_cls = FcBlock(feaVec_cls, 256, FLAGS.n_class, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

      # softmax layer to get loss
      with tf.variable_scope('softmax', reuse=reuse) as clsloss:
        sf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=feaVec_cls, labels=tf.one_hot(label, FLAGS.n_class))  # one_hot make the label same size as pred
        sf_loss = tf.reduce_mean(sf_loss)
      
      with tf.variable_scope('clsacc', reuse=reuse) as clsacc:
        sf_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(feaVec_cls, axis=1), label)))


    # reg branch
    with tf.variable_scope('reg', reuse=reuse) as reg:
      feav_list = feaMapFlatten(view_list)
      # FCN for each feature map in the view_list
      with tf.variable_scope('FCN', reuse=reuse) as cnBlock:
        reg_fea_list = []
        for i in xrange(num_views):
          reuse_reg = (i != 0)

          # reg_feaM has shape [batch_size, 4]
          var_reg, reg_feaM = FullyConnectedNet(feav_list[i], 4*4*256, FLAGS.weight_decay_ratio, reuse_reg, is_train, name='regFCN')

          reg_fea_list.append(reg_feaM)  # reg_feaM has shape [batch size, 4]

        reg_pred = feaVecConcat(reg_fea_list)  # the shape is NxVx4

      # L2 loss for regression
      with tf.variable_scope('l2loss', reuse=reuse) as regloss:
        reg_loss = tf.nn.l2_loss(tf.subtract(reg_pred, bbox))        
      
      # reg accuracy
      with tf.variable_scope('regacc', reuse=reuse) as regacc:
        valid = tf.cast(abs(reg_pred - bbox) < FLAGS.threshold_reg, tf.float32)
        reg_acc = tf.reduce_mean(valid)


    # depth branch
    with tf.variable_scope('depth', reuse=reuse) as dep:
      feaM_list = []
      for i in xrange(num_views):
        reuse_dep = (i != 0)

        # Deconv block to get the predicted feature map, output should be [N,128,128,1]
        var_dep, depth_pred_i = deConvBlocks(view_list[i], reuse_dep, is_train, name='depthDeconv')

        feaM_list.append(depth_pred_i)

      depth_pred = feaMapConcat(feaM_list)  # depth_pred has shape [N,V,128,128,1]

      # obtain the mask of object region in depth map
      mask_obj = tf.not_equal(depth_gt, tf.constant(1, dtype=tf.float32))
      effect_region = tf.reduce_sum(tf.where(mask_obj, tf.ones_like(depth_gt), tf.zeros_like(depth_gt)))

      # abs difference loss for depth
      with tf.variable_scope('depthLoss', reuse=reuse) as absDiffLoss:
        absDiffLoss = tf.reduce_sum(tf.where(
            mask_obj, 
            tf.losses.absolute_difference(depth_pred, depth_gt, reduction=tf.losses.Reduction.NONE), 
            tf.zeros_like(depth_gt)
          ))
        depth_loss = absDiffLoss / effect_region

      # depth accuracy
      with tf.variable_scope('depthacc', reuse=reuse) as depthacc:
        correct = tf.reduce_sum(tf.where(
            mask_obj, 
            tf.cast(abs(depth_pred - depth_gt) < FLAGS.threshold_depth, tf.float32),
            tf.zeros_like(depth_gt)
          ))
        depth_acc = correct / effect_region


  # combine loss
  comb_loss = sf_loss + reg_loss + depth_loss
  # obtain all variables of faster rcnn
  var_mvcnn = tf.contrib.framework.get_variables(mvcnn)

  return comb_loss, sf_loss, sf_acc, reg_loss, reg_acc, depth_loss, depth_acc, var_mvcnn


'''
  The train and test model
'''
def trainAndTest(dl_train, dl_test, ind_base):
  print 'The current base net is [' + net_label[ind_base] + ']'

  best_test_cls_loss, best_test_cls_acc = 100, 0
  best_test_reg_loss, best_test_reg_acc = 100, 0
  best_test_depth_loss, best_test_depth_acc = 100, 0

  image = tf.placeholder(tf.float32, [FLAGS.batch_size, 6, 128, 128, 3])  # x_image represents the input image
  label_gt = tf.placeholder(tf.int64, [FLAGS.batch_size])  # the ground truth label
  reg_gt = tf.placeholder(tf.float32, [FLAGS.batch_size, 6, 4]) # the bbox ground truth
  depth_gt = tf.placeholder(tf.float32, [FLAGS.batch_size, 6, 128, 128, 1]) # the bbox ground truth

  # faster rcnn model
  comb_loss, sf_loss, sf_acc, reg_loss, reg_acc, depth_loss, depth_acc, var_mvcnn = \
          MVCNN(image, label_gt, reg_gt, depth_gt, FLAGS.batch_size, None, True, ind_base)

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.005
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, dl_train.step, 1 - 10 ** (-FLAGS.iteration), staircase=True)

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_all)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(comb_loss, var_list=var_mvcnn, global_step=global_step)

  # array to store loss and accuracy
  list_loss_comb = np.zeros([FLAGS.iteration, 1])
  list_loss_cls, list_acc_cls = np.zeros([FLAGS.iteration, 1]), np.zeros([FLAGS.iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([FLAGS.iteration, 1]), np.zeros([FLAGS.iteration, 1])
  list_loss_depth, list_acc_depth = np.zeros([FLAGS.iteration, 1]), np.zeros([FLAGS.iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  epoch, time_count = 0, 0
  while (epoch < FLAGS.iteration):
    print '\n********************************* The {}th epoch training is processing *********************************'.format(epoch + 1)
    start_time = time.time()
    # random shuffle data
    view_epoch, depth_epoch, label_epoch, bbox_epoch, _, _ = dl_train.obatinEpochData(shuffle=True)

    # train for each epoch
    comb_loss_sum = 0
    cls_loss_sum, cls_acc_sum = 0, 0
    reg_loss_sum, reg_acc_sum = 0, 0
    depth_loss_sum, depth_acc_sum = 0, 0

    step_cur = 0
    while (step_cur < dl_train.step):
      # obtain current batch data for training
      view_batch = dl_train.obtainBatchData(view_epoch, step_cur)
      label_batch = dl_train.obtainBatchData(label_epoch, step_cur)
      bbox_batch = dl_train.obtainBatchData(bbox_epoch, step_cur)
      depth_batch = dl_train.obtainBatchData(depth_epoch, step_cur)
      
      # feed dict
      feed_dict_train = {image:view_batch, label_gt:label_batch, reg_gt:bbox_batch, depth_gt:depth_batch}

      # evaluation
      train_loss_cls = sf_loss.eval(feed_dict=feed_dict_train)
      train_acc_cls = sf_acc.eval(feed_dict=feed_dict_train)

      train_loss_reg = reg_loss.eval(feed_dict=feed_dict_train)
      train_acc_reg = reg_acc.eval(feed_dict=feed_dict_train)

      train_loss_depth = depth_loss.eval(feed_dict=feed_dict_train)
      train_acc_depth = depth_acc.eval(feed_dict=feed_dict_train)

      train_loss_comb = comb_loss.eval(feed_dict=feed_dict_train)

      # accumulate loss and accuracy of each step
      cls_loss_sum += train_loss_cls
      cls_acc_sum += train_acc_cls

      reg_loss_sum += train_loss_reg
      reg_acc_sum += train_acc_reg

      depth_loss_sum += train_loss_depth
      depth_acc_sum += train_acc_depth

      comb_loss_sum += train_loss_comb

      if step_cur % FLAGS.display_interval == 0:
        print('Train Epoch:{} [{}/{}]  Comb Loss: {:.8f}| Cls Loss: {:.8f}, Acc: {:.2f}%| Reg Loss: {:.8f}, Acc: {:.2f}%| Dep Loss: {:.8f}, Acc: {:.2f}%'.format(
          epoch + 1, (step_cur + 1) * FLAGS.batch_size , dl_train.step * FLAGS.batch_size,  
          train_loss_comb, train_loss_cls, 100. * train_acc_cls, train_loss_reg, 100. * train_acc_reg, train_loss_depth, 100. * train_acc_depth))


      # train the model
      train_step.run(feed_dict=feed_dict_train)
      step_cur += 1

    elapsed_time = time.time() - start_time
    time_count += elapsed_time
    # end of current epoch iteration 
    
    # evaluate the average loss and accuracy
    cls_loss_sum /= dl_train.step
    cls_acc_sum /= dl_train.step

    reg_loss_sum /= dl_train.step
    reg_acc_sum /= dl_train.step

    depth_loss_sum /= dl_train.step
    depth_acc_sum /= dl_train.step

    comb_loss_sum /= dl_train.step

    # print the result for each epoch
    print '\n********************************* The {}th epoch training has completed using {} s *********************************'.format(epoch + 1, elapsed_time)
    print 'Comb: Avg Loss is {:.8f}| Cls: Avg Loss is {:.8f}, Avg Acc is {:.2f}%| Reg: Avg Reg Loss is {:.8f}, Avg Acc is {:.2f}%| Dep: Avg Loss is {:.8f}, Avg Acc is {:.2f}%. \n'.format(
      comb_loss_sum, cls_loss_sum, 100 * cls_acc_sum, reg_loss_sum, 100 * reg_acc_sum, depth_loss_sum, 100 * depth_acc_sum)

    # store results
    list_loss_comb[epoch, 0] = comb_loss_sum
    list_loss_cls[epoch, 0], list_acc_cls[epoch, 0] = cls_loss_sum, cls_acc_sum
    list_loss_reg[epoch, 0], list_acc_reg[epoch, 0] = reg_loss_sum, reg_acc_sum
    list_loss_depth[epoch, 0], list_acc_depth[epoch, 0] = depth_loss_sum, depth_acc_sum


    '''
      test model
    '''
    test_cls_loss, test_cls_acc = 0, 0
    test_reg_loss, test_reg_acc = 0, 0
    test_dep_loss, test_dep_acc = 0, 0
    if (epoch + 1) % FLAGS.test_interval == 0 or (epoch + 1) == FLAGS.iteration:
      print '==========================================================================================================='
      print '--------------------------- [TEST] the trained model after {} epochs --------------------------------------'.format(epoch + 1)
      test_view_epoch, test_depth_epoch, test_label_epoch, test_bbox_epoch, _, _ = dl_test.obatinEpochData(shuffle=False)

      step_cur = 0
      while (step_cur < dl_test.step):
        # obtain current batch data for test
        test_view_batch = dl_test.obtainBatchData(test_view_epoch, step_cur)
        test_label_batch = dl_test.obtainBatchData(test_label_epoch, step_cur)
        test_bbox_batch = dl_test.obtainBatchData(test_bbox_epoch, step_cur)
        test_depth_batch = dl_test.obtainBatchData(test_depth_epoch, step_cur)
        
        # feed dict
        feed_dict_test = {image:test_view_batch, label_gt:test_label_batch, reg_gt:test_bbox_batch, depth_gt:test_depth_batch}
        
        # evaluation
        test_cls_loss += sf_loss.eval(feed_dict=feed_dict_test)
        test_cls_acc += sf_acc.eval(feed_dict=feed_dict_test)

        test_reg_loss += reg_loss.eval(feed_dict=feed_dict_test)
        test_reg_acc += reg_acc.eval(feed_dict=feed_dict_test)

        test_dep_loss += depth_loss.eval(feed_dict=feed_dict_test)
        test_dep_acc += depth_acc.eval(feed_dict=feed_dict_test)

        step_cur += 1

      # end test
      test_cls_loss /= dl_test.step
      test_cls_acc /= dl_test.step

      test_reg_loss /= dl_test.step
      test_reg_acc /= dl_test.step

      test_dep_loss /= dl_test.step
      test_dep_acc /= dl_test.step

      # store results
      if test_cls_acc > best_test_cls_acc:
        best_test_cls_acc = test_cls_acc
        best_test_cls_loss = test_cls_loss

      if test_reg_acc > best_test_reg_acc:
        best_test_reg_acc = test_reg_acc
        best_test_reg_loss = test_reg_loss

      if test_dep_acc > best_test_depth_acc:
        best_test_depth_acc = test_dep_acc
        best_test_depth_loss = test_dep_loss


      print 'The Test Cls Loss is {:.6f}, Acc is {:.2f}%, The [Best Acc] so far is {:.2f}%.'.format(test_cls_loss, 100 * test_cls_acc, 100 * best_test_cls_acc)
      print 'The Test Reg Loss is {:.6f}, Acc is {:.2f}%, The [Best Acc] so far is {:.2f}%.'.format(test_reg_loss, 100 * test_reg_acc, 100 * best_test_reg_acc)
      print 'The Test Dep Loss is {:.6f}, Acc is {:.2f}%, The [Best Acc] so far is {:.2f}%.'.format(test_dep_loss, 100 * test_dep_acc, 100 * best_test_depth_acc)
      print '==========================================================================================================='

      # save loss and accuracy
      np.save('res/MVCNN{}_test_cls.npy'.format(ind_base), [best_test_cls_loss, best_test_cls_acc])
      np.save('res/MVCNN{}_test_reg.npy'.format(ind_base), [best_test_reg_loss, best_test_reg_acc])
      np.save('res/MVCNN{}_test_depth.npy'.format(ind_base), [best_test_depth_loss, best_test_depth_acc])

      np.save('res/MVCNN{}_train_cls_loss.npy'.format(ind_base), list_loss_cls[:epoch + 1, :])
      np.save('res/MVCNN{}_train_cls_acc.npy'.format(ind_base), list_acc_cls[:epoch + 1, :])

      np.save('res/MVCNN{}_train_reg_loss.npy'.format(ind_base), list_loss_reg[:epoch + 1, :])
      np.save('res/MVCNN{}_train_reg_acc.npy'.format(ind_base), list_acc_reg[:epoch + 1, :])

      np.save('res/MVCNN{}_train_depth_loss.npy'.format(ind_base), list_loss_depth[:epoch + 1, :])
      np.save('res/MVCNN{}_train_depth_acc.npy'.format(ind_base), list_acc_depth[:epoch + 1, :])

      np.save('res/MVCNN{}_train_comb_loss.npy'.format(ind_base), list_loss_comb[:epoch + 1, :])


    # update epoch
    epoch += 1

  return time_count


# main code
def main(ind_base):
  dataLoader_train = dataLoader(FLAGS.dir_train, 1472, FLAGS.batch_size)
  dataLoader_test = dataLoader(FLAGS.dri_test, 448, FLAGS.batch_size)

  dataLoader_train.init_obj()
  dataLoader_test.init_obj()

  timeCost = trainAndTest(dataLoader_train, dataLoader_test, ind_base)
  print '\n=========== The training process has completed! Total [{} epochs] using [time {} s] ============'.format(FLAGS.iteration, timeCost)



if __name__ == "__main__":
  '''
    ind_base refers to different basenet structure
    - 0: ConvNet Block
    - 1: MobileNet Block
    - 2: ResNet Block
  '''
  ind_base = FLAGS.ind_base
  main(ind_base)

  '''
    Plot function
  '''
  # processPlot(ind_base)
