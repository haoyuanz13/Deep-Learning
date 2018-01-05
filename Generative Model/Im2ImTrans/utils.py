# from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
# import cv2
from PIL import Image
import pdb
import imageio, os

from functools import partial


'''
  Add noise
'''
def addNoise(x):
  x = x * np.random.randint(2, size=x.shape)
  x += np.random.randint(2, size=x.shape)

  return x



def merge(images, size):
  h, w, c = images.shape[1], images.shape[2], images.shape[-1]
  img = np.zeros((h * size[0], w * size[1]))

  N = size[0] * size[1]
  for idx, image in enumerate(images):
    if idx >= N:
      break

    i = idx % size[1]
    j = idx / size[1]

    img[j * h:j * h + h, i * w:i * w + w] = image

  return img


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

  total = loss_d.shape[0]

  processPlot_loss_GANs(path, total, loss_d, loss_g)


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
