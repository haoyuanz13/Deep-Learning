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
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("step", 100, "The step in each epoch iteration [100]")
flags.DEFINE_integer("iteration", 1000, "The max iteration times [1000]")
flags.DEFINE_integer("display_interval", 1, "The step interval to plot training loss and accuracy [10]")
flags.DEFINE_integer("test_interval", 100, "The step interval to test model [40]")
flags.DEFINE_integer("ind_base", 0, "The Conv block ind [0]")
flags.DEFINE_integer("n_class", 10, "The total number of class type [10]")
flags.DEFINE_float("weight_decay_ratio", 0.05, "weight decay ration [0.05]")
FLAGS = flags.FLAGS


net_label = ['ConvNet', 'MobileNet', 'BaseNet']


'''
  MVCNN 
  - Input views: NxVxWxHxC (N: batch size, V: number of views)
  - Input label: Nx1
'''
def MVCNN(views, label, bs, reuse, is_train, ind_base):
  with tf.variable_scope('MVCNN', reuse=reuse) as mvcnn:
    # obtain the number of views 
    num_views = views.get_shape().as_list()[1]
    input_channel = views.get_shape().as_list()[-1]

    # transpose views: [NxVxWxHxC] -> [VxNxWxHxC]
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    # multiple views concat
    with tf.variable_scope('multiView_concat', reuse=reuse) as mv:
      view_list = []
      for i in xrange(num_views):
        # set reuse True for i > 0 in order to share weight and bias
        reuse_base = (i != 0)

        raw_v = tf.gather(views, i)  # obtain raw view with shape [NxWxHxC]

        var_all, feaMap_i = BaseNet(raw_v, input_channel, FLAGS.weight_decay_ratio, ind_base, reuse_base, is_train, name='CNN1')

        # encode
        dim = np.prod(feaMap_i.get_shape().as_list()[1:])
        feaVec_i = tf.reshape(feaMap_i, [-1, dim])

        # append into feature map list for max pooling
        view_list.append(feaVec_i)

      # max pooling within multiple views 
      feaVec = viewPooling(view_list, 'viewPooling')

    # print "Concat feaVec shape: ", feaVec.get_shape().as_list()


    # fully connected layer: convert from 4x4x128 to 1024
    with tf.variable_scope('FC1', reuse=reuse) as fc1:
      feaVec = FcBlock(feaVec, 4*4*256, 1024, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

    # fully connected layer: convert from 1024 to 256
    with tf.variable_scope('FC2', reuse=reuse) as fc2:
      feaVec = FcBlock(feaVec, 1024, 256, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

    # fully connected layer: convert from 256 to 10
    with tf.variable_scope('FC3', reuse=reuse) as fc3:
      feaVec = FcBlock(feaVec, 256, FLAGS.n_class, is_train=is_train, reuse=reuse, wd=FLAGS.weight_decay_ratio)

    # softmax layer
    with tf.variable_scope('softmax', reuse=reuse) as sm:
      sf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=feaVec, labels=tf.one_hot(label, FLAGS.n_class))  # one_hot make the label same size as pred
      sf_loss = tf.reduce_mean(sf_loss)
      sf_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(feaVec, axis=1), label)))


  # obtain all variables of faster rcnn
  var_mvcnn = tf.contrib.framework.get_variables(mvcnn)

  return sf_loss, sf_acc, var_mvcnn



'''
  The train and test model
'''
def trainAndTest(dl_train, dl_test, ind_base):
  print 'The current base net is [' + net_label[ind_base] + ']'

  best_test_cls_loss, best_test_cls_acc = 100, 0


  image = tf.placeholder(tf.float32, [FLAGS.batch_size, 6, 128, 128, 3])  # x_image represents the input image
  label_gt = tf.placeholder(tf.int64, [FLAGS.batch_size])  # the ground truth label

  # faster rcnn model
  sf_loss, sf_acc, var_mvcnn = MVCNN(image, label_gt, FLAGS.batch_size, None, True, ind_base)

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.001
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, dl_train.step, 1 - 10 ** (-FLAGS.iteration), staircase=True)

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_all)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(sf_loss, var_list=var_mvcnn, global_step=global_step)

  list_loss_cls, list_acc_cls = np.zeros([FLAGS.iteration, 1]), np.zeros([FLAGS.iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  epoch, time_count = 0, 0
  while (epoch < FLAGS.iteration):
    print '\n********************************* The {}th epoch training is processing *********************************'.format(epoch + 1)
    start_time = time.time()
    # random shuffle data
    view_epoch, depth_epoch, label_epoch, bbox_epoch = dl_train.obatinEpochData(shuffle=True)

    # train for each epoch
    cls_loss_sum, cls_acc_sum = 0, 0

    step_cur = 0
    while (step_cur < dl_train.step):
      # obtain current batch data for training
      view_batch = dl_train.obtainBatchData(view_epoch, step_cur)
      label_batch = dl_train.obtainBatchData(label_epoch, step_cur).reshape([-1])

      # evaluation
      train_loss_cls = sf_loss.eval(feed_dict={image:view_batch, label_gt:label_batch})
      train_acc_cls = sf_acc.eval(feed_dict={image:view_batch, label_gt:label_batch})

      cls_loss_sum += train_loss_cls
      cls_acc_sum += train_acc_cls

      if step_cur % FLAGS.display_interval == 0:
        print('Train Epoch:{} [{}/{}]  Cls Loss: {:.8f}, Accuracy: {:.2f}%'.format(
          epoch + 1, (step_cur + 1) * FLAGS.batch_size , dl_train.step * FLAGS.batch_size, train_loss_cls, 100. * train_acc_cls))


      # train the model
      train_step.run(feed_dict={image:view_batch, label_gt:label_batch})
      step_cur += 1

    elapsed_time = time.time() - start_time
    time_count += elapsed_time
    # end of current epoch iteration 
    

    cls_loss_sum /= dl_train.step
    cls_acc_sum /= dl_train.step

    # print the result for each epoch
    print '\n********************************* The {}th epoch training has completed using {} s *********************************'.format(epoch + 1, elapsed_time)
    print 'The Avg Cls Loss is {:.8f}, Avg Acc is {:.2f}%. \n'.format(cls_loss_sum, 100 * cls_acc_sum)

    # store results
    list_loss_cls[epoch, 0], list_acc_cls[epoch, 0] = cls_loss_sum, cls_acc_sum


    '''
      test model
    '''
    test_cls_loss, test_cls_acc = 0, 0
    if (epoch + 1) % FLAGS.test_interval == 0:
      print '==========================================================================================================='
      print '--------------------------- [TEST] the trained model after {} epochs --------------------------------------'.format(epoch + 1)
      test_view_epoch, test_depth_epoch, test_label_epoch, test_bbox_epoch = dl_test.obatinEpochData(shuffle=False)

      step_cur = 0
      while (step_cur < dl_test.step):
        # obtain current batch data for test
        test_view_batch = dl_test.obtainBatchData(test_view_epoch, step_cur)
        test_label_batch = dl_test.obtainBatchData(test_label_epoch, step_cur).reshape([-1])

        # evaluation
        test_cls_loss += cls_loss.eval(feed_dict={image:test_view_batch, label_gt:test_label_batch})
        test_cls_acc += cls_acc.eval(feed_dict={image:test_view_batch, label_gt:test_label_batch})

        step_cur += 1

      # end test
      test_cls_loss /= dl_test.step
      test_cls_acc /= dl_test.step


      # store results
      if test_cls_acc > best_test_cls_acc:
        best_test_cls_acc = test_cls_acc
        best_test_cls_loss = test_cls_loss


      print 'The Test Cls Loss is {:.8f}, Acc is {:.2f}%, The [Best Acc] so far is {:.2f}%.'.format(test_cls_loss, 100 * test_cls_acc, 100 * best_test_cls_acc)
      print '==========================================================================================================='

      np.save('res/MVCNN{}_test_cls.npy'.format(ind_base), [best_test_cls_loss, best_test_cls_acc])
      np.save('res/MVCNN{}_train_cls_loss.npy'.format(ind_base), list_loss_cls[:epoch + 1, :])
      np.save('res/MVCNN{}_train_cls_acc.npy'.format(ind_base), list_acc_cls[:epoch + 1, :])


    # update epoch
    epoch += 1

    # store results
    np.save('res/MVCNN{}_test_cls.npy'.format(ind_base), [best_test_cls_loss, best_test_cls_acc])
    np.save('res/MVCNN{}_train_cls_loss.npy'.format(ind_base), list_loss_cls[:epoch, :])
    np.save('res/MVCNN{}_train_cls_acc.npy'.format(ind_base), list_acc_cls[:epoch, :])


  return time_count


# main code
def main(ind_base):
  dataLoader_train = dataLoader(FLAGS.dir_train, 192, FLAGS.batch_size)
  dataLoader_test = dataLoader(FLAGS.dri_test, 64, FLAGS.batch_size)

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