import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import utils as helper
import pdb

from layers_tf import *


def CnnClsReg(data, cls_label, reg_label, reg_lower, reg_upper, reuse):
  with tf.variable_scope('CCR', reuse=reuse) as ccr:
  
    # Common shared ConvNet with variable space: base
    with tf.variable_scope('base', reuse=reuse) as base:
      x_image = tf.reshape(data, [-1, 16, 16, 1])

      with tf.variable_scope('conv1', reuse=reuse):
        # 1st convolution layer
        x_conv1 = ConvNet(x_image, 16, 7, 1)
      
      with tf.variable_scope('conv2', reuse=reuse):
        # 2nd convolution layer
        x_conv2 = ConvNet(x_conv1, 8, 7, 1)

        # reshape image data from 4d into 2d
        x_conv2_2d = tf.reshape(x_conv2, [64, 4 * 4 * 8])

    # Cls branch with variable space: fc_cls
    with tf.variable_scope('fc_cls', reuse=reuse) as fc_cls:
      with tf.variable_scope('fc', reuse=reuse):
        # fully connected layer for cls
        x_fc_cls = FcNet(x_conv2_2d, 1)
        cls_pred = tf.sigmoid(x_fc_cls)

      with tf.variable_scope('loss', reuse=reuse):
        # compute cross entropy loss for cls
        cls_loss = CrossEntropy(cls_pred, cls_label)
        # compute accuracy
        cls_accuracy = tf.reduce_mean(tf.to_float(tf.equal( tf.cast(tf.greater(cls_pred, 0.5), tf.float32), cls_label )))

    # Reg branch with variable space: fc_reg
    with tf.variable_scope('fc_reg', reuse=reuse) as fc_reg:
      with tf.variable_scope('fc', reuse=reuse):
        # fully connected layer for regression
        x_fc_reg = FcNet(x_conv2_2d, 1)
        reg_pred = x_fc_reg
      
      with tf.variable_scope('loss', reuse=reuse):
        # compute L2 loss for reg
        reg_loss = L2Loss(reg_pred, reg_label)
        # compute accuracy
        valid = tf.multiply(tf.cast(tf.greater(reg_pred, reg_lower), tf.float32), tf.cast(tf.less(reg_pred, reg_upper), tf.float32))
        reg_accuracy = tf.reduce_mean(valid)
    

    var_base = tf.contrib.framework.get_variables(base)
    var_cls = tf.contrib.framework.get_variables(fc_cls)
    var_reg = tf.contrib.framework.get_variables(fc_reg)

  var_all = tf.contrib.framework.get_variables(ccr)
  return cls_loss, cls_accuracy, reg_loss, reg_accuracy, var_base, var_cls, var_reg, var_all


def train(im_train, cls_label, reg_label, reg_lower, reg_upper):
  x = tf.placeholder(tf.float32, [64, 16 * 16])

  cls_gt = tf.placeholder(tf.float32, [64, 1])
  reg_gt = tf.placeholder(tf.float32, [64, 1])
  reg_l = tf.placeholder(tf.float32, [64, 1])
  reg_u = tf.placeholder(tf.float32, [64, 1])

  cls_loss, cls_acc, reg_loss, reg_acc, var_base, var_cls, var_reg, var_all = CnnClsReg(x, cls_gt, reg_gt, reg_l, reg_u, True)

  # handle optimization process
  loss = tf.add(cls_loss, tf.scalar_mul(0.01, reg_loss))


  opt_base = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=var_all)
  # opt_cls = tf.train.GradientDescentOptimizer(0.1).minimize(cls_loss, var_list=[var_base ,var_cls])
  # opt_reg = tf.train.GradientDescentOptimizer(0.001).minimize(reg_loss, var_list=[var_base ,var_reg])

  # opt = tf.group(opt_cls, opt_reg)
  # opt = tf.group(opt, opt_reg)

  # define optimizer
  
  # opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=var_all)

  i, iteration = 0, 10000
  list_loss_cls, list_acc_cls = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([iteration, 1]), np.zeros([iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  while (i < iteration):
    # training accuracy
    train_loss_cls = cls_loss.eval(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})
    train_accuacy_cls = cls_acc.eval(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})

    train_loss_reg = reg_loss.eval(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})
    train_accuacy_reg = reg_acc.eval(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})

    print ('***************** The {}th iteration *******************'.format(i + 1))
    print ('--The cls loss {:.5f} and cls accuracy {:.5f}%.'.format(train_loss_cls, train_accuacy_cls * 100))
    print ('--The reg loss {:.5f} and reg accuracy {:.5f}%. \n'.format(train_loss_reg, train_accuacy_reg * 100))

    list_loss_cls[i, 0], list_acc_cls[i, 0] = train_loss_cls, train_accuacy_cls
    list_loss_reg[i, 0], list_acc_reg[i, 0] = train_loss_reg, train_accuacy_reg

    if train_accuacy_reg == 1:
      break

    opt_base.run(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})
    # opt_cls.run(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})
    # opt_reg.run(feed_dict={x:im_train, cls_gt:cls_label, reg_gt:reg_label, reg_l:reg_lower, reg_u:reg_upper})
    
    i += 1

  return list_loss_cls, list_acc_cls, list_loss_reg, list_acc_reg, i


# main code
def main():
  # load data
  folder = "datasets/detection"
  train_npy = "detection_imgs.npy"
  cls_label_npy = "detection_labs.npy"
  reg_label_npy = "detection_width.npy"

  # the shape of training image is 64 x 16 x 16
  im_train = helper.dataloader(folder, train_npy).reshape([64, 16*16])

  # corresponding class label (64, 1)
  cls_label = helper.dataloader(folder, cls_label_npy).reshape(-1, 64).transpose()
  # corresponding region ground truth (64, 1)
  reg_label = helper.dataloader(folder, reg_label_npy).reshape(-1, 64).transpose()

  reg_upper, reg_lower = reg_label + 0.5, reg_label - 0.5

  # train model
  list_loss_cls, list_acc_cls, list_loss_reg, list_acc_reg, stop_ind = train(im_train, cls_label, reg_label, reg_lower, reg_upper)


  title_cls, title_reg = "Loss & Accuracy - Iteration (Cls Branch)", "Loss & Accuracy - Iteration (Reg Branch)"  
  # plot loss and accuracy
  helper.processPlot(stop_ind, list_loss_cls[:stop_ind, :], list_acc_cls[:stop_ind, :], title_cls)
  helper.processPlot(stop_ind, list_loss_reg[:stop_ind, :], list_acc_reg[:stop_ind, :], title_reg)


if __name__ == "__main__":
  main()