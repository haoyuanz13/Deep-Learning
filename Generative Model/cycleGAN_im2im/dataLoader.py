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
  Convert image to three channel if necessary
'''
def checkChannel(img):
  if len(img.shape) != 3:
    im_cur_3c = np.zeros([FLAGS.fine_height, FLAGS.fine_width, 3]).astype(np.float32)
    im_cur_3c[:, :, 0] = img
    im_cur_3c[:, :, 1] = img
    im_cur_3c[:, :, 2] = img
    return im_cur_3c

  else:
    return img



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

  # check image dimension
  # img_A = checkChannel(img_A)
  # img_B = checkChannel(img_B)

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



'''
  data process for cufs_students faces
'''
def dataProcess_cufs_students():
  # imgs = []
  path = "data/cufs_students/"

  folder_name_sketch =["sketches_train", "sketches_test"]
  folder_name_photo =["photos_train", "photos_test"]

  valid_images = [".jpg",".gif",".png",".tga"]
  
  # pre-process sketches data
  for file_name in folder_name_sketch:
    path_cur = path + file_name + "/"
    is_train = (file_name == "sketches_train")

    # read in images
    imgs = []
    for f in sorted(os.listdir(path_cur)):
      ext = os.path.splitext(f)[1]
      if ext.lower() not in valid_images:
          continue
      
      # im_cur = np.array(Image.open(os.path.join(path, f))).astype(np.float32)
      im_cur = io.imread(os.path.join(path_cur, f))

      '''
        resize test data but random crop training data
      '''
      if is_train:
        im_cur = randomCrop(im_cur, flip=True)
      
      else:
        # resize
        im_cur = scipy.misc.imresize(im_cur, [FLAGS.fine_height, FLAGS.fine_width])

      im_cur = im_cur.astype(np.float32)
      # normalize data
      # im_cur = normData(im_cur, twoSides=True)
      im_cur = im_cur / 127.5 - 1

      im_cur_3c = np.zeros([FLAGS.fine_height, FLAGS.fine_width, 3]).astype(np.float32)
      if len(im_cur.shape) != 3:
        im_cur_3c[:, :, 0] = im_cur
        im_cur_3c[:, :, 1] = im_cur
        im_cur_3c[:, :, 2] = im_cur
      
      else:
        im_cur_3c = im_cur

      imgs.append(im_cur_3c)

    # store data
    print '======> saving the [' + file_name + '] file set..................'
    np.save(('data/cufs_students/numpyData/' + file_name + '.npy'), imgs)


  # pre-process photo data
  for file_name in folder_name_photo:
    path_cur = path + file_name + "/"
    is_train = (file_name == "photos_train")

    # read in images
    imgs = []
    for f in sorted(os.listdir(path_cur)):
      ext = os.path.splitext(f)[1]
      if ext.lower() not in valid_images:
          continue
      
      # im_cur = np.array(Image.open(os.path.join(path, f))).astype(np.float32)
      im_cur = io.imread(os.path.join(path_cur, f))

      if is_train:
        im_cur = randomCrop(im_cur, flip=True)
      else:
        im_cur = scipy.misc.imresize(im_cur, [FLAGS.fine_height, FLAGS.fine_width])

      im_cur = im_cur.astype(np.float32)
      # normalize data
      # im_cur = normData(im_cur, twoSides=True)
      im_cur = im_cur / 127.5 - 1

      imgs.append(im_cur)

    # store data
    print '======> saving the [' + file_name + '] file set..................'
    np.save(('data/cufs_students/numpyData/' + file_name + '.npy'), imgs)


'''
  data loader for cufs_students faces
'''
def dataloader_cufs_students():
  print "\n========== CUFS Studenst Dataset (.npy files) =========="
  base_path = "data/cufs_students/numpyData/"
  file_name = ["sketches_train", "sketches_test", "sketches_val", "photos_train", "photos_test", "photos_val"]

  sketches_dict, photos_dict = \
          {'train': [], 'test': [], 'val': []}, {'train': [], 'test': [], 'val': []}

  print '===>> loading [' + file_name[0] + '] dataset .......'
  sketches_dict['train'] = np.load(base_path + file_name[0] + ".npy")

  
  skt_test = np.load(base_path + file_name[1] + ".npy")
  print '===>> loading [' + file_name[1] + '] dataset .......'
  sketches_dict['test'] = skt_test[:50, :, :, :]
  
  print '===>> loading [' + file_name[2] + '] dataset .......'
  sketches_dict['val'] = skt_test[50:, :, :, :]


  print '===>> loading [' + file_name[3] + '] dataset .......'
  photos_dict['train'] = np.load(base_path + file_name[3] + ".npy")

  pht_test = np.load(base_path + file_name[4] + ".npy")
  print '===>> loading [' + file_name[4] + '] dataset .......'
  photos_dict['test'] = pht_test[:50, :, :, :]
  
  print '===>> loading [' + file_name[5] + '] dataset .......'
  photos_dict['val'] = pht_test[50:, :, :, :]

  return sketches_dict, photos_dict


'''
  Change the data format: left part is photo and right part is sketch
'''
def connectData(is_train=True):
  # imgs = []
  path = "data/cufs_students/"

  if is_train:
    folder_name =["photos_train", "sketches_train"]
  
  else:
    folder_name =["photos_test", "sketches_test"]

  valid_images = [".jpg",".gif",".png",".tga"]
  

  data_all = []
  # pre-process training data
  for file_name in folder_name:
    print "The current folder is [" + file_name + "]"
    path_cur = path + file_name + "/"

    count = 0
    # read in images
    for f in sorted(os.listdir(path_cur)):
      ext = os.path.splitext(f)[1]
      if ext.lower() not in valid_images:
          continue

      im_cur = io.imread(os.path.join(path_cur, f))
      im_cur = im_cur.astype(np.float32)

      # resize
      im_cur = scipy.misc.imresize(im_cur, [FLAGS.fine_height, FLAGS.fine_width])

      # normalize data
      # im_cur = normData(im_cur, twoSides=True)
      # im_cur = im_cur / 127.5 - 1

      im_cur_3c = np.zeros([FLAGS.fine_height, FLAGS.fine_width, 3]).astype(np.float32)
      if len(im_cur.shape) != 3:
        im_cur_3c[:, :, 0] = im_cur
        im_cur_3c[:, :, 1] = im_cur
        im_cur_3c[:, :, 2] = im_cur
      
      else:
        im_cur_3c = im_cur


      if file_name == "photos_train" or file_name == "photos_test":
        im_concat = np.zeros([FLAGS.fine_height, 2 * FLAGS.fine_width, 3])
        im_concat[:, 0:FLAGS.fine_width, :] = im_cur_3c
        data_all.append(im_concat)

      else:
        im_concat = data_all[count]
        im_concat[:, FLAGS.fine_width:, :] = im_cur_3c
        data_all[count] = im_concat

        count += 1

  # store data
  total = len(data_all)
  save_path = "data/cufs_std_concat/train" if is_train else "data/cufs_std_concat/test"
  for i in xrange(total):
    scipy.misc.imsave(save_path + ('/{}.jpg'.format(i+1)), data_all[i])




if __name__ == '__main__':
  # connectData(is_train=False)
  dataProcess_cufs_students()
  # dataloader_cufs_students()

