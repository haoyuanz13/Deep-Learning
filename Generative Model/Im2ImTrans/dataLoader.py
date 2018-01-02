import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import pdb
from PIL import Image
import skimage.io as io 


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
  dataloader
'''
def dataProcess_cufs(readTest=False):
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

      # normalize
      img = normData(img, twoSides=False)
      # expand dimension to make it as shape [img_height, img_width, 1]
      img = np.expand_dims(img, axis=2)

      # image data
      dict_cur['img'].append(img)

      # img label
      dict_cur['order'].append(int(file_info[1]))

  
  dict_cur['img'] = np.array(dict_cur['img'])
  dict_cur['order'] = np.array(dict_cur['order'])

  return dict_cur


'''
  load cufs data
'''
def dataloader_cufs():
  print "======================== CUFS Dataset ========================"
  # load cufs data
  train_dict_cufs = dataProcess_cufs()
  test_dict_cufs = dataProcess_cufs(True)

  train_imgs_cufs, train_ind_cufs = train_dict_cufs['img'], train_dict_cufs['order']
  test_imgs_cufs, test_ind_cufs = test_dict_cufs['img'], test_dict_cufs['order']

  return train_imgs_cufs, train_ind_cufs, test_imgs_cufs, test_ind_cufs


'''
  data process for celeba faces
'''
def dataProcess_celeba():
  imgs = []
  path = "data/celeba/imgs"
  valid_images = [".jpg",".gif",".png",".tga"]
  
  count, file_ind = 0, 0
  for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    
    # im_cur = np.array(Image.open(os.path.join(path, f))).astype(np.float32)
    im_cur = io.imread(os.path.join(path, f))
    im_cur = im_cur.astype(np.float32)

    # normalize data
    im_cur = normData(im_cur, twoSides=False)
    im_cur = np.expand_dims(im_cur, axis=2)
    imgs.append(im_cur)

    count += 1
    if count == 10000:
      print '======> saving the {} file set..................'.format(file_ind)
      np.save(('data/celeba/numpyData/imgSet1_{}.npy'.format(file_ind)), imgs)
      file_ind += 1
      count = 0
      imgs[:] = []


'''
  data loader for celeba faces
'''
def dataloader_celeba(ind, twoSides=True):
  print "======================== Celeba Dataset ========================"
  print ('>>>>>>>>>>> load training data set {} .................'.format(ind))
  if twoSides:
    path = ("data/celeba/numpyData/imgSet_{}.npy".format(ind))
  else:
    path = ("data/celeba/numpyData/imgSet1_{}.npy".format(ind))

  imgs = np.load(path)  
  return imgs


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

    # read in images
    imgs = []
    for f in sorted(os.listdir(path_cur)):
      ext = os.path.splitext(f)[1]
      if ext.lower() not in valid_images:
          continue
      
      # im_cur = np.array(Image.open(os.path.join(path, f))).astype(np.float32)
      im_cur = io.imread(os.path.join(path_cur, f))
      im_cur = im_cur.astype(np.float32)

      # normalize data
      im_cur = normData(im_cur, twoSides=True)
      
      if len(im_cur.shape) == 3:
        im_cur = im_cur[:, :, 0]
      
      im_cur = np.expand_dims(im_cur, axis=2)
      imgs.append(im_cur)

    # store data
    print '======> saving the [' + file_name + '] file set..................'
    np.save(('data/cufs_students/numpyData/' + file_name + '.npy'), imgs)

  # pre-process photo data
  for file_name in folder_name_photo:
    path_cur = path + file_name + "/"

    # read in images
    imgs = []
    for f in sorted(os.listdir(path_cur)):
      ext = os.path.splitext(f)[1]
      if ext.lower() not in valid_images:
          continue
      
      # im_cur = np.array(Image.open(os.path.join(path, f))).astype(np.float32)
      im_cur = io.imread(os.path.join(path_cur, f))
      im_cur = im_cur.astype(np.float32)

      # normalize data
      im_cur = normData(im_cur, twoSides=True)
      imgs.append(im_cur)

    # store data
    print '======> saving the [' + file_name + '] file set..................'
    np.save(('data/cufs_students/numpyData/' + file_name + '.npy'), imgs)


'''
  data loader for cufs_students faces
'''
def dataloader_cufs_students():
  print "======================== CUFS Studenst Dataset ========================"
  base_path = "data/cufs_students/numpyData/"
  file_name = ["sketches_train", "sketches_test", "photos_train", "photos_test"]

  sketches_dict, photos_dict = {'train': [], 'test': []}, {'train': [], 'test': []}

  print '>>>>>>>>>>> load [' + file_name[0] + '] dataset .................'
  sketches_dict['train'] = np.load(base_path + file_name[0] + ".npy")

  print '>>>>>>>>>>> load [' + file_name[1] + '] dataset .................'
  sketches_dict['test'] = np.load(base_path + file_name[1] + ".npy")

  print '>>>>>>>>>>> load [' + file_name[2] + '] dataset .................'
  photos_dict['train'] = np.load(base_path + file_name[2] + ".npy")

  print '>>>>>>>>>>> load [' + file_name[3] + '] dataset .................'
  photos_dict['test'] = np.load(base_path + file_name[3] + ".npy")

  return sketches_dict, photos_dict


if __name__ == '__main__':
  # dataloader_cufs()
  # dataProcess_celeba()
  # dataloader_celeba(3, twoSides=False)

  # dataProcess_cufs_students()
  dataloader_cufs_students()

