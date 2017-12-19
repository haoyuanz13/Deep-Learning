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
  
