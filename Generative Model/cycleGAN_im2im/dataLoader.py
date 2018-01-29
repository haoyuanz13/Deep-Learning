import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import pdb
from PIL import Image
import scipy.misc
import skimage.io as io 

from utils import *

################
# Define flags #
################
flags = tf.app.flags
flags.DEFINE_integer("fine_height", 256, "The target height of resized image [256]")
flags.DEFINE_integer("fine_width", 256, "The target width of resized image [256]")
flags.DEFINE_integer("load_height", 286, "The load height of training image [286]")
flags.DEFINE_integer("load_width", 286, "The load width of training image [286]")
FLAGS = flags.FLAGS


'''
  normalize into [-1, 1] or [0, 1]
'''
def normData(arr, twoSides=True):
  max_x, min_x = np.max(arr), np.min(arr)
  # normalize to [-1, 1]
  if twoSides:
    arr = 2 * ((arr - min_x) / (max_x - min_x)) - 1
  # normalize to [0, 1]
  else:
    arr = (arr - min_x) / (max_x - min_x)

  return arr


'''
  load concat data
  - Input image_path: store file directory for image A[0] and B[1]
'''
def load_data(image_path, flip=True, is_test=False):
  img_A = load_image(image_path[0])
  img_B = load_image(image_path[1])

  img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

  img_A = img_A / 127.5 - 1.
  img_B = img_B / 127.5 - 1.

  img_AB = np.concatenate((img_A, img_B), axis=2)  # A is photo [0:3]; B is sketch [3:6]
  # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
  return img_AB


'''
  separate images
'''
def load_image(image_path, is_grayscale=False):
  if (is_grayscale):
    input_img = scipy.misc.imread(image_path, flatten=True).astype(np.float)
  else:
    input_img = scipy.misc.imread(image_path, mode='RGB').astype(np.float)

  return input_img
