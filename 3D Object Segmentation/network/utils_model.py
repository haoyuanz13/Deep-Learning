'''
  File name: utils.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains helper functions such as the viewPooling
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pdb

import matplotlib.pyplot as plt
from PIL import Image


'''
  reshape feature maps into feature vector
'''
def feaMapFlatten(view_maps):
  view_features = []
  # encode feature map into feature vector
  for fea_m in view_maps:
    dim = np.prod(fea_m.get_shape().as_list()[1:])
    fea_v = tf.reshape(fea_m, [-1, dim])
    view_features.append(fea_v)

  return view_features  


'''
  view pooling to compress multiple feature maps
  - Input view_features: a list of feature maps from multiple views
  - Output vp: one feature map with the same shape as each single feature map
'''
def viewPooling(view_features, name):
  # expand shape dimension eg. [h, w, c] -> [1, h, w, c]
  vp = tf.expand_dims(view_features[0], 0)

  for v in view_features[1:]:
    v = tf.expand_dims(v, 0)
    vp = tf.concat([vp, v], 0)

  # max pooling in the first dimension
  out = tf.reduce_max(vp, [0], name=name)

  # print name, " | out shape: ", out.get_shape().as_list()

  return out

'''
  feature vector concat
  - Input view_features: a list of feature matrix each of them has shape [NxK]
  - Output out: a tensor with shape [NxVxK]
'''
def feaVecConcat(view_features):
  vp = tf.expand_dims(view_features[0], 0)

  for v in view_features[1:]:
    v = tf.expand_dims(v, 0)
    vp = tf.concat([vp, v], 0)

  # transpose
  out = tf.transpose(vp, perm=[1, 0, 2])
  return out


'''
  feature map concat
'''
def feaMapConcat(view_features):
  vp = tf.expand_dims(view_features[0], 0)

  for v in view_features[1:]:
    v = tf.expand_dims(v, 0)
    vp = tf.concat([vp, v], 0)

  # transpose
  out = tf.transpose(vp, perm=[1, 0, 2, 3, 4])
  return out


'''
  L2 loss
'''
def L2Loss(y_pred, y_gt):
  diff = tf.square(tf.subtract(y_pred, y_gt))
  loss = tf.reduce_mean(diff)

  return loss

'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_acc_loss(iteration, loss, accuracy, test_loss, test_acc, title, takeAvg=True, showAcc=True):
  fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))

  x = np.arange(0, iteration, 1)

  if takeAvg:
    new_loss = np.zeros_like(loss)
    new_acc = np.zeros_like(accuracy)

    for i in xrange(iteration):
      new_loss[i] = np.mean(loss[:i+1])
      new_acc[i] = np.mean(accuracy[:i+1])

    loss = new_loss
    accuracy = new_acc


  Ax0.plot(x, loss)
  # Ax0.text(0.5, 80, , fontsize=12)
  Ax0.text(0.95, 0.01, 'The average test loss is {:.4f}'.format(test_loss),
        verticalalignment='bottom', horizontalalignment='right', transform=Ax0.transAxes,
        color='red', fontsize=15)

  Ax0.set_title('Loss Value') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('loss')
  Ax0.grid(True)
  

  Ax1.plot(x, accuracy)
  Ax1.set_xlabel('iteration times')
  Ax1.grid(True)

  if showAcc: 
    Ax1.text(0.95, 0.01, 'The average test accuracy is {:.4f}%'.format(test_acc * 100),
          verticalalignment='bottom', horizontalalignment='right', transform=Ax1.transAxes,
          color='red', fontsize=15)
    Ax1.set_title('Prediction Accuracy')
    Ax1.set_ylabel('Accuracy')

  else:
    Ax1.text(0.95, 0.01, 'The average test diff distance is {:.4f}'.format(test_acc),
          verticalalignment='bottom', horizontalalignment='right', transform=Ax1.transAxes,
          color='red', fontsize=15)
    Ax1.set_title('Prediction Diff Distance')
    Ax1.set_ylabel('Diff Distance')


  plt.suptitle(title, fontsize=16)
  plt.show()


'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot(ind_base):
  test_res_cls = np.load('res/MVCNN{}_test_cls.npy'.format(ind_base))
  test_res_reg = np.load('res/MVCNN{}_test_reg.npy'.format(ind_base))
  test_res_depth = np.load('res/MVCNN{}_test_depth.npy'.format(ind_base)) 

  train_cls_loss = np.load('res/MVCNN{}_train_cls_loss.npy'.format(ind_base))
  train_cls_acc = np.load('res/MVCNN{}_train_cls_acc.npy'.format(ind_base))
  train_reg_loss = np.load('res/MVCNN{}_train_reg_loss.npy'.format(ind_base))
  train_reg_acc = np.load('res/MVCNN{}_train_reg_acc.npy'.format(ind_base))
  train_depth_loss = np.load('res/MVCNN{}_train_depth_loss.npy'.format(ind_base))
  train_depth_acc = np.load('res/MVCNN{}_train_depth_acc.npy'.format(ind_base))

  total = train_cls_loss.shape[0]
  title = 'Cls Branch: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_cls_loss, train_cls_acc, test_res_cls[0], test_res_cls[1], title)

  title = 'Reg Branch: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_reg_loss, train_reg_acc, test_res_reg[0], test_res_reg[1], title, showAcc=False)

  title = 'Depth Branch: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_depth_loss, train_depth_acc, test_res_depth[0], test_res_depth[1], title, showAcc=False)


if __name__ == '__main__':
  processPlot(0)
  
