import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from PIL import Image

from layers import *
from helper import *
from RPN_section import *
from spatial_transformer import *

classes_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
  load data as dictionary
'''
def dataProcess(readTest=False):
  dict_cur = {'img': [], 'cls_gt': [], 'reg_gt': [], 'label': []}
  
  # obtain data path (test or train)
  if readTest:
    print '>>>>>>>>>>> load training data .................'
    path = 'cifar10_transformed/devkit/test.txt'
  else:
    print '>>>>>>>>>>> load testing data .................'
    path = 'cifar10_transformed/devkit/train.txt'
  
  with open(path) as f:
    for line in f:
      file_info = line.split()
      file_name = file_info[0]

      image = 'cifar10_transformed/imgs/' + file_name
      mask_image = 'cifar10_transformed/masks/' + file_name

      # image data
      dict_cur['img'].append(mpimg.imread(image))

      # mask ground truth
      mask_cur = np.array(Image.open(mask_image))
      dict_cur['cls_gt'].append(mask_cur)

      # region ground truth
      x_y_w_map = np.zeros((6, 6, 3), dtype=np.float32)
      x_y_w_map[mask_cur == 1, 0] = file_info[2]
      x_y_w_map[mask_cur == 1, 1] = file_info[3]
      x_y_w_map[mask_cur == 1, 2] = file_info[4]
      dict_cur['reg_gt'].append(x_y_w_map)

      # class label from 0 to 9 inclusive
      dict_cur['label'].append(int(file_info[1]))
  
  dict_cur['img'] = np.array(dict_cur['img'])
  dict_cur['cls_gt'] = np.array(dict_cur['cls_gt'])
  dict_cur['reg_gt'] = np.array(dict_cur['reg_gt'])
  dict_cur['label'] = np.array(dict_cur['label'])

  return dict_cur

'''
  Faster RCNN (RPN + FC for classification)
'''
def fasterRCNN(data, mask_batch, reg_batch, label, bs, reuse):
  with tf.variable_scope('fatserRCNN', reuse=reuse) as fasterrcnn:
    # Common shared ConvNet with variable space: BaseNet
    with tf.variable_scope('BaseNet', reuse=reuse) as base:
      # 1st convolution layer
      with tf.variable_scope('conv1', reuse=reuse):
        data = ConvBlock(data, 3, 32, 5, 1, is_train=True, reuse=reuse)
        data_conv1 = maxPool(data)

      # the 2nd convolution layer
      with tf.variable_scope('conv2', reuse=reuse):
        data_conv1 = ConvBlock(data_conv1, 32, 64, 5, 1, is_train=True, reuse=reuse)
        data_conv2 = maxPool(data_conv1)

      # the 3rd convolution layer
      with tf.variable_scope('conv3', reuse=reuse):
        data_conv2 = ConvBlock(data_conv2, 64, 128, 5, 1, is_train=True, reuse=reuse)
        data_conv3 = maxPool(data_conv2)

      # the 3rd convolution layer
      with tf.variable_scope('conv4', reuse=reuse):
        data_conv4 = ConvBlock(data_conv3, 128, 256, 3, 1, is_train=True, reuse=reuse)

      # intermediate layer
      with tf.variable_scope('intermediate', reuse=reuse):
        inter_map = ConvBlock(data_conv4, 256, 256, 3, 1, is_train=True, reuse=reuse)


    # Cls branch with variable space: cls_branch
    with tf.variable_scope('cls_branch', reuse=reuse) as cls_branch:
      # get cls feature map
      with tf.variable_scope('clsMap', reuse=reuse):
        Weight = tf.Variable(tf.truncated_normal([1, 1, 256, 1], stddev=0.1), name='W_cls')
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name='b_cls')
        cls_map = conv2d(inter_map, Weight) + bias
        cls_map = tf.reshape(cls_map, [-1, 6, 6])

      # compute cls loss (sigmoid cross entropy)
      with tf.variable_scope('clsLoss', reuse=reuse):
        # obtain valid places (pos and neg) in mask
        cond_cls = tf.not_equal(mask_batch, tf.constant(2, dtype=tf.float32))
        # compute the sigmoid cross entropy loss: choose loss where cond is 1 while select 0
        cross_entropy_classify = tf.reduce_sum(
          tf.where(cond_cls, tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_batch, logits=cls_map), tf.zeros_like(mask_batch)))
        # count the pos and neg numbers
        effect_area_cls = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cross_entropy_classify /= effect_area_cls

      # compute cls accuracy
      with tf.variable_scope('clsAcc', reuse=reuse):
        correct = tf.reduce_sum(tf.where(cond_cls, tf.cast(abs(mask_batch - tf.nn.sigmoid(cls_map)) < 0.5, tf.float32), tf.zeros_like(mask_batch)))
        effect_area = tf.reduce_sum(tf.where(cond_cls, tf.ones_like(mask_batch), tf.zeros_like(mask_batch)))
        cls_acc = correct / effect_area


    # Reg branch with variable space: reg_branch
    with tf.variable_scope('reg_branch', reuse=reuse) as reg_branch:
      # get reg feature map
      with tf.variable_scope('regMap', reuse=reuse):
        Weight = tf.Variable(tf.truncated_normal([1, 1, 256, 3], stddev=0.1), name='W_reg')
        bias = tf.Variable(tf.constant([24., 24., 32.]), name='b_reg')
        reg_map = conv2d(inter_map, Weight) + bias

      # compute reg loss (smooth l1 loss)
      with tf.variable_scope('regLoss', reuse=reuse):
        # normalize coordinates
        reg_map_norm = tf.stack([tf.divide(reg_map[:, :, :, 0], 32), tf.divide(reg_map[:, :, :, 1], 32), tf.log(reg_map[:, :, :, 2])])
        reg_map_norm = tf.transpose(reg_map_norm, [1, 2, 3, 0])
    
        reg_mask_norm = tf.stack([tf.divide(reg_batch[:, :, :, 0], 32), tf.divide(reg_batch[:, :, :, 1], 32), tf.log(reg_batch[:, :, :, 2])])
        reg_mask_norm = tf.transpose(reg_mask_norm, [1, 2, 3, 0])

        cond_reg = tf.not_equal(reg_batch, tf.constant(0, dtype=tf.float32))
        smooth_L1_loss_reg = tf.reduce_sum(
            tf.where(cond_reg, tf.losses.huber_loss(reg_map_norm, reg_mask_norm, reduction=tf.losses.Reduction.NONE), tf.zeros_like(reg_mask_norm)))

        effect_area_reg = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_mask_norm), tf.zeros_like(reg_mask_norm)))
        smooth_L1_loss_reg /= effect_area_reg

      # compute reg accuracy
      with tf.variable_scope('regAcc', reuse=reuse):
        correct = tf.reduce_sum(tf.where(cond_reg, tf.cast(abs(reg_batch - reg_map) < 1, tf.float32), tf.zeros_like(reg_batch)))
        effect_area = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_batch), tf.zeros_like(reg_batch)))
        reg_acc = correct / effect_area

    
    # spatial transform layer
    with tf.variable_scope('STN', reuse=reuse) as stn:
      feaMap_cls_sig = tf.nn.sigmoid(cls_map)

      # find the max index
      cls_map_flat = tf.reshape(feaMap_cls_sig, [bs, 36])  # convert into [bs, 36]
      obj_ind = tf.reshape(tf.argmax(cls_map_flat, axis=1), [bs, 1])  # max ind in each mask
      ind_batch = np.arange(bs).reshape([-1, bs]).transpose()  # ind to label images

      # concat indices 
      ind_cat = tf.concat([ind_batch, obj_ind], 1)

      # use gather_nd to obtain x, y and w for each max location
      reg_map_flat = tf.reshape(reg_map, [bs, 36, 3])  # convert into [bs, 36, 3]
      xyw = tf.gather_nd(reg_map_flat, ind_cat)  # xyw is ndarray type

      # stack to create the theta
      theta_1, theta_3 = tf.zeros_like(xyw[:, 0]), tf.zeros_like(xyw[:, 0])
      theta_0, theta_4 = tf.divide(xyw[:, 2], 48), tf.divide(xyw[:, 2], 48)
      theta_2, theta_5 = tf.divide(tf.subtract(xyw[:, 0], 24), 24), tf.divide(tf.subtract(xyw[:, 1], 24), 24)

      theta_batch = tf.stack([theta_0, theta_1, theta_2, theta_3, theta_4, theta_5], axis=1)  # theta has shape [batch size, 6]

      # spatial transformation: outputs should be [batch size, 4, 4, 256]
      spatial_map = transformer(data_conv4, theta_batch, (4, 4))

    # fully connected layer: convert from 4x4x256 to 256
    with tf.variable_scope('FC1', reuse=reuse) as fc1:
      fc_map = tf.reshape(spatial_map, [bs, 4*4*256])
      fc_map_flat = FcBlock(fc_map, 4*4*256, 256, is_train=True, reuse=reuse)

    # the addtional convolution layer
    # with tf.variable_scope('conv_add', reuse=reuse) as conv_add:
    #   W = tf.get_variable('weights', [4, 4, 256, 256], initializer=tf.contrib.layers.variance_scaling_initializer())
    #   b = tf.get_variable('biases', [1, 1, 1, 256], initializer=tf.constant_initializer(0.0))
    #   fc_map = tf.nn.conv2d(spatial_map, W, strides=[1, 1, 1, 1], padding='VALID') + b
    #   fc_map = tf.contrib.layers.batch_norm(fc_map, is_training=True, scale=True, fused=True, updates_collections=None)
    #   fc_map = tf.nn.relu(fc_map)

    # fully connected layer: convert from 256 to 10
    with tf.variable_scope('FC', reuse=reuse) as fc:
      # fc_map_flat = tf.reshape(fc_map, [bs, 256])
      W = tf.get_variable('weights', [256, 10], initializer=tf.contrib.layers.variance_scaling_initializer())
      pred = tf.matmul(fc_map_flat, W)

    # softmax layer
    with tf.variable_scope('softmax', reuse=reuse) as sm:
      sf_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.one_hot(label, 10))  # one_hot make the label same size as pred
      sf_loss = tf.reduce_mean(sf_loss)
      sf_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred, axis=1), label)))

  # joint loss    
  comb_loss = 5 * smooth_L1_loss_reg + cross_entropy_classify + sf_loss
  # obtain all variables of faster rcnn
  var_fastrcnn = tf.contrib.framework.get_variables(fasterrcnn)

  return comb_loss, sf_loss, sf_acc, cross_entropy_classify, cls_acc, smooth_L1_loss_reg, reg_acc, var_fastrcnn


'''
  The train and test model
'''
def trainAndTest(dict_train, dict_test):
  epoch, batch_size, step, iteration = 0, 100, 100, 1000
  display_interval, test_interval = 10, 50

  image = tf.placeholder(tf.float32, [batch_size, 48, 48, 3])  # x_image represents the input image
  mask_gt = tf.placeholder(tf.float32, [batch_size, 6, 6])  # mask for classification
  reg_gt = tf.placeholder(tf.float32, [batch_size, 6, 6, 3])  # the ground truth for proposal regression
  label_gt = tf.placeholder(tf.int64, [batch_size])  # the ground truth label

  comb_loss, sf_loss, sf_acc, cls_loss, cls_acc, reg_loss, reg_acc, var_frcnn = \
                                      fasterRCNN(image, mask_gt, reg_gt, label_gt, batch_size, None)

  # define learning rate decay parameters
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.001
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, step, 1 - 10 ** (-iteration), staircase=True)

  # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_all)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(comb_loss, var_list=var_frcnn, global_step=global_step)

  list_loss_rcnn, list_acc_rcnn = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_cls, list_acc_cls = np.zeros([iteration, 1]), np.zeros([iteration, 1])
  list_loss_reg, list_acc_reg = np.zeros([iteration, 1]), np.zeros([iteration, 1])

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  while (epoch < iteration):
    print '\n********************************* The {}th epoch training is processing *********************************'.format(epoch + 1)
    # data shuffle 
    arr = np.arange(dict_train['img'].shape[0])
    np.random.shuffle(arr)

    # batch size data
    data_epoch = dict_train['img'][arr[:], :, :, :]
    mask_epoch = dict_train['cls_gt'][arr[:], :, :]
    reg_mask_epoch = dict_train['reg_gt'][arr[:], :, :, :]
    label_epoch = dict_train['label'][arr[:]]

    # train for each epoch
    cls_loss_sum, reg_loss_sum, rcnn_loss_sum = 0, 0, 0
    cls_acc_sum, reg_acc_sum, rcnn_acc_sum = 0, 0, 0

    step_cur = 0
    while (step_cur < step):
      # obtain current batch data for training
      data_batch = data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
      mask_batch = mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
      reg_mask_batch = reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
      label_batch = label_epoch[batch_size*step_cur : batch_size*(step_cur+1)]

      # evaluation
      train_loss_rcnn = sf_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_rcnn = sf_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      train_loss_cls = cls_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_cls = cls_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      train_loss_reg = reg_loss.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      train_acc_reg = reg_acc.eval(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})

      rcnn_loss_sum += train_loss_rcnn
      cls_loss_sum += train_loss_cls
      reg_loss_sum += train_loss_reg

      rcnn_acc_sum += train_acc_rcnn
      cls_acc_sum += train_acc_cls
      reg_acc_sum += train_acc_reg

      if step_cur % display_interval == 0:
        print('Train Epoch:{} [{}/{}]  Obj Loss: {:.8f}, Accuracy: {:.2f}% || Cls Loss: {:.8f}, Accuracy: {:.2f}% || Reg Loss: {:.8f}, Accuracy: {:.4f}%'.format(
          epoch + 1, step_cur * batch_size , step * batch_size, train_loss_rcnn, 100 * train_acc_rcnn, train_loss_cls, 100. * train_acc_cls, train_loss_reg, 100. * train_acc_reg))

      # train the model
      train_step.run(feed_dict={image:data_batch, mask_gt:mask_batch, reg_gt:reg_mask_batch, label_gt:label_batch})
      step_cur += 1

    # end of current epoch iteration 
    
    rcnn_loss_sum /= step
    cls_loss_sum /= step
    reg_loss_sum /= step

    rcnn_acc_sum /= step
    cls_acc_sum /= step
    reg_acc_sum /= step
    # print the result for each epoch
    print '\n********************************* The {}th epoch training has completed *********************************'.format(epoch + 1)
    print 'The Avg Obj Loss is {:.8f}, Avg Acc is {:.2f}% || The Avg Cls Loss is {:.8f}, Avg Acc is {:.2f}% || The Avg Reg Loss is {:.8f}, Avg Acc is {:.2f}%. \n'.format(
      rcnn_loss_sum, 100 * rcnn_acc_sum, cls_loss_sum, 100 * cls_acc_sum, reg_loss_sum, 100 * reg_acc_sum)

    # store results
    list_loss_rcnn[epoch, 0], list_acc_rcnn[epoch, 0] = rcnn_loss_sum, rcnn_acc_sum
    list_loss_cls[epoch, 0], list_acc_cls[epoch, 0] = cls_loss_sum, cls_acc_sum
    list_loss_reg[epoch, 0], list_acc_reg[epoch, 0] = reg_loss_sum, reg_acc_sum


    '''
      test model
    '''
    test_rcnn_loss, test_rcnn_acc = 0, 0
    test_cls_loss, test_cls_acc, test_reg_loss, test_reg_acc = 0, 0, 0, 0
    if (epoch + 1) % test_interval == 0:
      print '==========================================================================================================='
      print '--------------------------- [TEST] the trained model after {} epochs --------------------------------------'.format(epoch + 1)
      test_data_epoch = dict_test['img']
      test_mask_epoch = dict_test['cls_gt']
      test_reg_mask_epoch = dict_test['reg_gt']
      test_label_epoch = dict_test['label']

      step_cur = 0
      while (step_cur < step):
        # obtain current batch data for training
        test_data_batch = test_data_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
        test_mask_batch = test_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :]
        test_reg_mask_batch = test_reg_mask_epoch[batch_size*step_cur : batch_size*(step_cur+1), :, :, :]
        test_label_batch = test_label_epoch[batch_size*step_cur : batch_size*(step_cur+1)]

        # evaluation
        test_rcnn_loss += sf_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_rcnn_acc += sf_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        
        test_cls_loss += cls_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_cls_acc += cls_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})

        test_reg_loss += reg_loss.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})
        test_reg_acc += reg_acc.eval(feed_dict={image:test_data_batch, mask_gt:test_mask_batch, reg_gt:test_reg_mask_batch, label_gt:test_label_batch})

        step_cur += 1

      # end test
      test_rcnn_loss /= step
      test_rcnn_acc /= step
      test_cls_loss /= step
      test_cls_acc /= step
      test_reg_loss /= step
      test_reg_acc /= step

      print 'The Test Obj Loss is {:.8f}, Acc is {:.2f}% || The Test Cls Loss is {:.8f}, Acc is {:.2f}% || The Test Reg Loss is {:.8f}, Acc is {:.2f}%.'.format(
        test_rcnn_loss, 100 * test_rcnn_acc, test_cls_loss, 100 * test_cls_acc, test_reg_loss, 100 * test_reg_acc)
      print '==========================================================================================================='

      # store results
      np.save('res/FasterRCNN_test_rcnn.npy', [test_rcnn_loss, test_rcnn_acc])
      np.save('res/FasterRCNN_test_cls.npy', [test_cls_loss, test_cls_acc])
      np.save('res/FasterRCNN_test_reg.npy', [test_reg_loss, test_reg_acc])

      np.save('res/FasterRCNN_train_rcnn_loss.npy', list_loss_rcnn[:epoch + 1, :])
      np.save('res/FasterRCNN_train_rcnn_acc.npy', list_acc_rcnn[:epoch + 1, :])
      np.save('res/FasterRCNN_train_cls_loss.npy', list_loss_cls[:epoch + 1, :])
      np.save('res/FasterRCNN_train_cls_acc.npy', list_acc_cls[:epoch + 1, :])
      np.save('res/FasterRCNN_train_reg_loss.npy', list_loss_reg[:epoch + 1, :])
      np.save('res/FasterRCNN_train_reg_acc.npy', list_acc_reg[:epoch + 1, :])

    # update epoch
    epoch += 1

    # store results
    np.save('res/FasterRCNN_test_rcnn.npy', [test_rcnn_loss, test_rcnn_acc])
    np.save('res/FasterRCNN_test_cls.npy', [test_cls_loss, test_cls_acc])
    np.save('res/FasterRCNN_test_reg.npy', [test_reg_loss, test_reg_acc])

    np.save('res/FasterRCNN_train_rcnn_loss.npy', list_loss_rcnn[:epoch, :])
    np.save('res/FasterRCNN_train_rcnn_acc.npy', list_acc_rcnn[:epoch, :])
    np.save('res/FasterRCNN_train_cls_loss.npy', list_loss_cls[:epoch, :])
    np.save('res/FasterRCNN_train_cls_acc.npy', list_acc_cls[:epoch, :])
    np.save('res/FasterRCNN_train_reg_loss.npy', list_loss_reg[:epoch, :])
    np.save('res/FasterRCNN_train_reg_acc.npy', list_acc_reg[:epoch, :])

  return


# main code
def main(showRes):
  if not showRes:
    # load data
    dict_train = dataProcess(readTest=False)
    dict_test = dataProcess(readTest=True)
    trainAndTest(dict_train, dict_test)

  else:
    processPlot()


if __name__ == "__main__":
  op = True
  main(op)