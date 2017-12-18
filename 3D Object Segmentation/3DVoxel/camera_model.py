import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
#from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting

import h5py
import pdb 
from scipy import interpolate


# 1D vector to rgb values, provided by ../input/plot3d.py
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

# Transform data from 1d to 3d rgb
def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)

def rot_x(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [1,  0,   0],
                  [0, cos, -sin], 
                  [0, sin,  cos] 
                  ])


def rot_y(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [cos, 0, sin],
                  [0,   1,  0], 
                  [-sin,0, cos] 
                  ])

def rot_z(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [cos, -sin, 0],
                  [sin,  cos, 0], 
                  [0  ,  0,   1] 
                  ])


class CamModel(object):
  def __init__(self, img_size=(128, 128), N=60):
    # Camera constants   
    self.img_h, self.img_w = img_size 

    # Intrinsic parameters
    self.K = np.array([ [200, 0, 64], 
                   [0, 200, 64], 
                   [0, 0, 1]])


    self.T = np.array([30, 30, 400]).reshape([3,1]) 
    xx, yy, zz = np.meshgrid(range(N), range(N), range(N)) 
    xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten() 
    ones = np.ones_like(xx)
    self.xyz_ones = np.vstack([xx, yy, zz, ones])
  
    # INF depth (mean we don't know the depth)
    self.INF = 1e3

  def project_3d_2d(self, data_sample, pose=(0,0,0), visual=False): 
    print('\n=====> Pose:', pose)
    theta_x, theta_y, theta_z = pose 
    img_h, img_w = self.img_h, self.img_w 

    R_x = rot_x(theta_x/180.*np.pi)
    R_y = rot_y(theta_y/180.*np.pi)
    R_z = rot_z(theta_z/180.*np.pi)

    R = np.dot(np.dot(R_z, R_y), R_x)
    Rt = np.concatenate([R, self.T], 1)  
    P = (np.dot(self.K, Rt))
    # print P

    # x, y, z 
    uvs = np.dot(np.dot(self.K, Rt), self.xyz_ones)

    # Depth map 
    obj_depth_map = uvs[2] 

    # Normalize the last row to get uv_ones 
    uv_ones = uvs*1.0/(uvs[2]) 

    # Floor indices 
    uu = np.floor(uv_ones[0])
    vv = np.floor(uv_ones[1])

    # Find max and min to get bbox 
    min_u, max_u = np.min(uu), np.max(uu)
    min_v, max_v = np.min(vv), np.max(vv)
    bbox = [min_v, max_v, min_u, max_u]

    print('==> Bbox:', bbox)
 
    two_D_img = np.zeros([img_h, img_w, 3])
    three_D_img = np.ones([img_h, img_w]) * self.INF 
   
    for i in range(img_h):
        uu_cond = (np.abs(uu-i)<=0.2)
        for j in range(img_w):
            vv_cond = (np.abs(vv-j)<=0.2)
            index = np.where(uu_cond*vv_cond)[0]
            if len(index):
              # Find color 
              two_D_img[i,j] = np.mean(data_sample[index])*255 
              # Find depth 
              three_D_img[i, j] = np.mean(obj_depth_map[index])
       
    two_D_img = np.clip(two_D_img, 0, 255) 
    two_D_img[np.where(two_D_img < 5)] = 255 
    print('Depth:', three_D_img)

 
    if visual:
      plt.subplot(121)
      plt.imshow(two_D_img.astype(np.uint8))
      plt.title('2D')
      plt.subplot(122)
      plt.imshow(three_D_img.astype(np.uint8))
      plt.title('Depth Map')
      plt.show() 

    return two_D_img,  three_D_img, bbox
   
 

  def test_project_3d_2d(self):
    # Load data 
    data_path = './data/3d-mnist/'
    with h5py.File(data_path + 'full_dataset_vectors.h5', 'r') as hf:
      # 1e4 training samples, each of which are flattened voxel (16*16*16)
      # each element in 4096-D vector is intensity, ranging from 0 or 1  
      x_train_raw = hf["X_train"][:] # (10000, 4096) 
      y_train_raw = hf["y_train"][:] #(10000,) 

    # Check length of the dataset 
    assert(len(x_train_raw) == len(y_train_raw))


    # Each data point now has shape 16x16x16x3 (rgb) 
    x_train = rgb_data_transform(x_train_raw)
    data_sample = x_train[0,:,:,:,0]
    self.project_3d_2d(data_sample) 

