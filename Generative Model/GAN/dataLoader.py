import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import pdb


'''
  dataloader
'''
def dataProcess(readTest=False):
  dict_cur = {'img': [], 'order': []}
  
  # obtain data path (test or train)
  if readTest:
    print '>>>>>>>>>>> load testing data .................'
    path = 'data/cufs/devkit/test.txt'
  else:
    print '>>>>>>>>>>> load training data .................'
    path = 'data/cufs/devkit/train.txt'
  
  with open(path) as f:
    for line in f:
      file_info = line.split()
      file_name = file_info[0]

      image_path = 'data/cufs/imgs/' + file_name
      img = mpimg.imread(image_path)

      # expand dimension to make it as shape [img_height, img_width, 1]
      img = np.expand_dims(img, axis=2)

      # image data
      dict_cur['img'].append(img)

      # img label
      dict_cur['order'].append(int(file_info[1]))

  
  dict_cur['img'] = np.array(dict_cur['img'])
  dict_cur['order'] = np.array(dict_cur['order'])

  return dict_cur