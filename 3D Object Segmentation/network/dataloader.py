'''
  File name: dataloader.py
  Author: Haoyuan Zhang
  Date: 12/16/2017
'''

'''
  The file contains function to load and pre-process data
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import glob

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.image as mpimg
import pdb
from PIL import Image
import time
import math


'''
  Global variables
'''
dirtxt = "data_new/"
dir2D = "data_new/2D/"
dir3D = "data_new/3D/"

img_h, img_w = 128, 128

pose_type = ['pose_0', 'pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5']
txt_name = ['train_pose_0_upto_2500.txt', 'train_pose_1_upto_2500.txt', 'train_pose_2_upto_2500.txt', 
            'train_pose_3_upto_2500.txt', 'train_pose_4_upto_2500.txt', 'train_pose_5_upto_2500.txt']

dri_train = "dataset/train/"
dri_test = "dataset/test/"

'''
  Pre-process data for each pose
  - Input N: the number of instance
  - Input ind_V: represents the pose type
'''
def preProcess_Pose(N=100, ind_V=0):
  # store 2d views and depth image
  views, depth = np.zeros((N, img_h, img_w, 3)), np.zeros((N, img_h, img_w, 1))

  # class label and bbox label
  label, bbox = np.zeros(N), np.zeros((N, 4))

  path_txt = dirtxt + txt_name[ind_V]
  path_2d = dir2D + pose_type[ind_V]
  path_3d = dir3D + pose_type[ind_V]

  ind = 0
  with open(path_txt) as f:
    for line in f:
      print "Processing the {}th instance.............".format(ind)
      file_info = line.split()

      path2d = path_2d + "/" + file_info[0] + ".npy"
      view_cur = np.load(path2d)


      path3d = path_3d + "/" + file_info[0] + ".npy"
      depth_cur = np.load(path3d)
      depth_cur = np.expand_dims(depth_cur, axis=2)

      # store data
      views[ind, :, :, :] = view_cur
      depth[ind, :, :, :] = depth_cur
      label[ind] = int(file_info[1])
      bbox[ind, 0], bbox[ind, 1] = float(file_info[2]), float(file_info[3])
      bbox[ind, 2], bbox[ind, 3] = float(file_info[4]), float(file_info[5])

      ind += 1
      if ind >= N:
        break

  return views, depth, label, bbox 



'''
  Pre-process data
  - Output view_set: NxVxHxWx3 (N:number of samples; V:number of views; HxWx3:single sample size)
  - Output depth_set: NxVxHxWx1 (N:number of samples; V:number of views; HxWx1:single sample size)
  - Output label_set: Nx1 (N:number of samples; V:number of views; HxWxC:single sample size)
  - Output view_set: NxVxHxWxC (N:number of samples; V:number of views; HxWxC:single sample size)
'''
def preProcess(N=100, V=6, trainSet=True):
  start_pose = 0
  views_set, depth_set = np.zeros((N, V, img_h, img_w, 3)), np.zeros((N, V, img_h, img_w, 1))
  label_set, bbox_set = np.zeros(N), np.zeros((N, V, 4))

  while start_pose < 6:
    print "\nProcessing the {}th type pose camera data..........................".format(start_pose)
    view_cur, depth_cur, label_cur, bbox_cur = preProcess_Pose(N, start_pose)

    views_set[:, start_pose, :, :, :] = view_cur
    depth_set[:, start_pose, :, :, :] = depth_cur
    bbox_set[:, start_pose, :] = bbox_cur

    if start_pose == 0:
      label_set = label_cur

    start_pose += 1

  # save data
  dri_store = dri_train if trainSet else dri_test
  print "Saving data.........................."
  np.save(dri_store + "views.npy", views_set)
  np.save(dri_store + "depths.npy", depth_set)
  np.save(dri_store + "labels.npy", label_set)
  np.save(dri_store + "bboxes.npy", bbox_set)
  print "=====>>> Pre-process data is done !"


'''
  Class dataloader
'''
class dataLoader(object):
  def __init__(self, path, N=1000, bs=100):
    self.views = None
    self.depth = None
    self.label = None
    self.bbox = None

    self.bboxDepth = None
    self.bboxCenterRegion = None

    self.dir = path

    self.N = N
    self.batch_size = bs
    self.step = 0

  # constructor
  def init_obj(self):
    view_path = self.dir + "views.npy"
    dataset_view = np.load(view_path)

    depth_path = self.dir + "depths.npy"
    dataset_depth = np.load(depth_path)

    label_path = self.dir + "labels.npy"
    dataset_label = np.load(label_path)

    bbox_path = self.dir + "bboxes.npy"
    dataset_bbox = np.load(bbox_path)

    bboxDepth_path = self.dir + "bboxDepth.npy"
    dataset_bboxDepth = np.load(bboxDepth_path)

    bboxCenterRegion_path = self.dir + "bboxCenterRegion.npy"
    dataset_bboxCenReg = np.load(bboxCenterRegion_path)

    self.views = dataset_view[:self.N, :, :, :, :]
    self.depth = dataset_depth[:self.N, :, :, :, :]
    self.label = dataset_label[:self.N]
    self.bbox = dataset_bbox[:self.N, :, :]
    self.bboxDepth = dataset_bboxDepth[:self.N, :, :, :, :]
    self.bboxCenterRegion = dataset_bboxCenReg[:self.N, :, :]

    self.normalize()
    self.step = self.N / self.batch_size;

  # obtain data for one epoch
  def obatinEpochData(self, shuffle=False):
    ind = np.arange(self.N)
    if shuffle:
      np.random.shuffle(ind)

    view_epoch = self.views[ind[:], :, :, :, :]
    depth_epoch = self.depth[ind[:], :, :, :, :]
    label_epoch = self.label[ind[:]]
    bbox_epoch = self.bbox[ind[:], :, :]

    bboxdep_epoch = self.bboxDepth[ind[:], :, :, :, :]
    bboxCenReg_epoch = self.bboxCenterRegion[ind[:], :, :]

    return view_epoch, depth_epoch, label_epoch, bbox_epoch, bboxdep_epoch, bboxCenReg_epoch

  # obtain batch data
  def obtainBatchData(self, dataset, ind_step):
    dim_num = len(dataset.shape)
    if ind_step >= self.step or ind_step < 0:
      print "ERROR: cannot access batch !"
      return None

    if dim_num == 5:
      return dataset[ind_step * self.batch_size : (ind_step + 1) * self.batch_size, :, :, :, :]

    if dim_num == 1:
      return dataset[ind_step * self.batch_size : (ind_step + 1) * self.batch_size]

    if dim_num == 3:
      return dataset[ind_step * self.batch_size : (ind_step + 1) * self.batch_size, :, :]


  # normalize data: bbox and depth image
  def normalize(self):
    img_h, img_w = 128, 128
    self.bbox[:, :, 0:2] = (self.bbox[:, :, 0:2] - img_w / 2) / float(img_w)
    self.bbox[:, :, 2:4] = (self.bbox[:, :, 2:4] - img_h / 2) / float(img_h)

    self.depth = self.depth / float(1000)
    self.bboxDepth = self.bboxDepth / float(1000)



if __name__ == "__main__":
  # preProcess(500, trainSet=False)
  N = 100
  data = dataLoader(dri_test, N, 10)
  data.init_obj()

  view, depth, label, bbox, bboxDep, bboxCenReg = data.obatinEpochData()

  # i = 0
  # bbox_depth = np.zeros((N, 6, 20, 20, 1))
  # bbox_center_region = np.zeros((N, 6, 4))  # [r_min, r_max, c_min, c_max]

  # while i < 6:
  #   print 'Pre-processing on the view {}.......................'.format(i)
  #   depth_cur = depth[:, i, :, :, 0]
  
  #   bbox_cur_raw = bbox[:, i, :]

  #   bbox_cur_raw[:, 0:2] = bbox_cur_raw[:, 0:2] * 128 + 64
  #   bbox_cur_raw[:, 2:4] = bbox_cur_raw[:, 2:4] * 128 + 64

  #   r_center = ((bbox_cur_raw[:, 2] + bbox_cur_raw[:, 3]) / 2).astype(np.int16)
  #   c_center = ((bbox_cur_raw[:, 0] + bbox_cur_raw[:, 1]) / 2).astype(np.int16)

  #   r_min, r_max = r_center - 10, r_center + 10
  #   c_min, c_max = c_center - 10, c_center + 10

  #   bbox_center_region[:, i, 0] = r_min
  #   bbox_center_region[:, i, 1] = r_max
  #   bbox_center_region[:, i, 2] = c_min
  #   bbox_center_region[:, i, 3] = c_max

  #   j = 0
  #   while j < N:
  #     print "---> Store bbox depth image for the {}th instance.......".format(j)
  #     bbox_depth[j, i, :, :, 0] = depth_cur[j, r_min[j]:r_max[j], c_min[j]:c_max[j]]
  #     j += 1  

      
  #   i += 1

  # np.save(dri_test + 'bboxDepth.npy', bbox_depth)
  # np.save(dri_test + 'bboxCenterRegion.npy', bbox_center_region)

  pdb.set_trace()

  label_batch = data.obtainBatchData(label, 2)
  
  im = view[1, 0, :, :, 0]
  dep = depth[1, 0, :, :, 0]

  fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))
  Ax0.imshow(im)  

  Ax1.imshow(dep)
  plt.show()
