from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import math
import json
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import random
import pprint
import scipy.misc
from time import gmtime, strftime
from PIL import Image
import pdb
import imageio, os

from functools import partial


pp = pprint.PrettyPrinter()
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


'''
  Add noise
'''
def addNoise(x):
  x = x * np.random.randint(2, size=x.shape)
  x += np.random.randint(2, size=x.shape)
  return x


'''
  merge images
'''
def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    
    img[j * h : j * h + h, i * w : i * w + w, :] = image

  return img 

'''
  recover images
'''
def inverse_transform(images):
  return (images + 1.) / 2.


'''
  preprocess data to constraint certain size and data augmentuation
'''
def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
  # test data
  if is_test:
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
  
  # train data
  else:
    img_A = scipy.misc.imresize(img_A, [load_size, load_size])
    img_B = scipy.misc.imresize(img_B, [load_size, load_size])

    # random crop
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
    img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

    # random crop
    if flip and np.random.random() > 0.5:
      img_A = np.fliplr(img_A)
      img_B = np.fliplr(img_B)

  return img_A, img_B



'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_loss_GANs(path, iteration, d_loss, g_loss):
  fig, (Ax0, Ax1) = plt.subplots(2, 1, figsize = (8, 20))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, d_loss)
  Ax0.set_title('Model D Loss vs Iterations') 
  Ax0.set_ylabel('loss value')
  Ax0.grid(True)
  

  Ax1.plot(x, g_loss)
  Ax1.set_title('Model G Loss vs Iterations')
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('loss value')
  Ax1.grid(True)

  plt.show()
  
  fig.savefig(path + '/loss_curve.png')   # save the figure to file
  plt.close(fig)    # close the figure


'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot_GANs():
  path = 'im2im_res/curves'
  loss_d = np.load(path + '/loss_d.npy')
  loss_g = np.load(path + '/loss_g.npy')

  # total = loss_d.shape[0]
  total = 450

  processPlot_loss_GANs(path, total, loss_d[:total], loss_g[:total])


'''
  generate gif
'''
def gifGenerate_GANs():
  path = 'im2im_res/samples'
  imgs = []
  for filename in sorted(os.listdir(path)):
    print (filename)
    imgs.append(imageio.imread(path + '/' + filename))

  imageio.mimsave(path + '/samples.gif', imgs)



if __name__ == "__main__":
  processPlot_GANs()
