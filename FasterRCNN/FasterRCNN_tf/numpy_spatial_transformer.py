# Simple version of spatial_transformer.py 
# Only one image and one channel 
import numpy as np 
import cv2 
import pdb 
import matplotlib.pyplot as plt 

###############################################################
# Changable parameter
# scale_H:# The indices of the grid of the target output is
# scaled to [-1, 1]. Set False to stay in normal mode 
SCALE_H = False

def _meshgrid(height, width, scale_H = SCALE_H):
  if scale_H:
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                    np.linspace(-1, 1, height))
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

  else:
    x_t, y_t = np.meshgrid(range(0,width), range(0,height))
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
# print '--grid size:', grid.shape 
  return grid 


def _interpolate(im, x, y, out_size, scale_H = SCALE_H):
  # constants
  height = im.shape[0]
  width =  im.shape[1]

  height_f = float(height)
  width_f =  float(width)
  out_height = out_size[0]
  out_width = out_size[1]
  zero = np.zeros([], dtype='int32')
  max_y = im.shape[0] - 1
  max_x = im.shape[1] - 1

  if scale_H:
    # # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0) * (width_f) / 2.0
    y = (y + 1.0) * (height_f) / 2.0

  # do sampling
  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1

  # print 'x0:', y0 
  # print 'x1:', y1 
  # Limit the size of the output image 
  x0 = np.clip(x0, zero, max_x)
  x1 = np.clip(x1, zero, max_x)
  y0 = np.clip(y0, zero, max_y)
  y1 = np.clip(y1, zero, max_y)
  

  Ia = im[y0, x0]
  Ib = im[y1, x0]
  Ic = im[y0, x1]
  Id = im[y1, x1]

  # print
  # plt.figure(2)
  # plt.subplot(221)
  # plt.imshow(Ia)
  # plt.subplot(222)
  # plt.imshow(Ib)
  # plt.subplot(223)
  # cv2.imshow('Ic', Ic)
  # plt.subplot(224)
  # plt.imshow(Id)
  # cv2.waitKey(0)

  wa = (x1 - x) * (y1 - y)
  wb = (x1 - x) * (y - y0)
  wc = (x - x0) * (y1 - y)
  wd = (x - x0) * (y - y0)
  # print 'wabcd...', wa,wb, wc,wd 

  out = wa * Ia + wb * Ib + wc * Ic + wd * Id
  # print '--shape of out:', out.shape
  return out 

def _transform(theta, input_dim, out_size):
  height, width = input_dim.shape[0], input_dim.shape[1]
  theta = np.reshape(theta, (3, 3))
  # print '--Theta:', theta 
  # print '-- Theta shape:', theta.shape  

  # grid of (x_t, y_t, 1), eq (1) in ref [1]
  out_height = out_size[0]
  out_width = out_size[1]
  grid = _meshgrid(out_height, out_width)

  # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
  T_g = np.dot(theta, grid)
  x_s = T_g[0,:]
  y_s = T_g[1,:]
  t_s = T_g[2,:]
  # print '-- T_g:', T_g 
  # print '-- x_s:', x_s 
  # print '-- y_s:', y_s
  # print '-- t_s:', t_s

  t_s_flat = np.reshape(t_s, [-1])
  # Ty changed 
  # x_s_flat = np.reshape(x_s, [-1])
  # y_s_flat = np.reshape(y_s, [-1])
  x_s_flat = np.reshape(x_s, [-1])/t_s_flat
  y_s_flat = np.reshape(y_s, [-1])/t_s_flat
  

  input_transformed =  _interpolate(input_dim, x_s_flat, y_s_flat, out_size) 

  output = np.reshape(input_transformed, [out_height, out_width])
  # output = output.astype(np.int32)
  return output


def numpy_transformer(img, H, out_size, scale_H = SCALE_H): 
  h, w = img.shape[0], img.shape[1]
  # Matrix M 
  M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

  if scale_H:
    H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
    # print 'H_transformed:', H_transformed 
    # img2 = _transform(H_transformed, img, [h,w])
    img2 = _transform(H_transformed, img, out_size)
  else:
    # img2 = _transform(np.linalg.inv(H), img, [h,w])
    img2 = _transform(np.linalg.inv(H), img, out_size)
  return img2 



# demo code for transformer test
def test_transformer(img, H, scale_H = SCALE_H): 
  # img = cv2.imread('figure_1-1.png',0)
  h, w = img.shape[0], img.shape[1]
  print '-- h, w:', h, w 

  print 'Affine matrix is: \n', H
  # Apply homography transformation 

  out_size = [32, 32]

  # img2 = cv2.warpPerspective(img, H, (32, 32))
  # img3 = numpy_transformer(img, H, out_size)

  # print '-- Reprojection error is {:.6f}:'.format(np.mean(np.square(img3 - img2)))
  # Reprojection = abs(img3 - img2)
  
  img2 = np.zeros([out_size[0], out_size[1], 3]).astype(np.int8)
  img3 = np.zeros([out_size[0], out_size[1], 3]).astype(np.int8)
  Reprojection = np.zeros([out_size[0], out_size[1], 3])

  # apply the same pipeline for all three channels
  for i in [0, 1, 2]:
    img2[:, :, i] = cv2.warpPerspective(img[:, :, i], H, (32, 32))
    img3[:, :, i] = numpy_transformer(img[:, :, i], H, out_size)

    print '-- Reprojection error of channel {} is {:.6f}:'.format(i, np.mean(np.square(img3[:, :, i] - img2[:, :, i])))
    Reprojection[:, :, i] = abs(img3[:, :, i] - img2[:, :, i])

  # plot transformed image
  try:
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')

    plt.subplot(222)
    plt.imshow(img2, cmap='gray')
    plt.title('cv2.warpPerspective')

    plt.subplot(223)
    plt.imshow(img3, cmap='gray')
    plt.title('Transformer')

    plt.subplot(224)
    plt.imshow(Reprojection, cmap='gray')
    plt.title('Reprojection Error')
    plt.show()

  except KeyboardInterrupt:
    plt.close()
    exit(1)

  # return img3


if __name__ == "__main__":
  print "----> loading training images and masks .............................."
  train_imgs = np.load("dataRPN/train_imgs.npy")
  train_masks = np.load("dataRPN/train_mask.npy")
  print "----> Completed! \n"

  print "----> loading test images and masks .................................."
  test_imgs = np.load("dataRPN/test_imgs.npy")
  test_masks = np.load("dataRPN/test_mask.npy")
  print "----> Completed! \n"

  print "----> loading training and test regression ground truth .............."
  # train_reg = np.load("dataRPN/train_reg.npy")
  # test_reg = np.load("dataRPN/test_reg.npy")

  train_reg = np.load("dataRPN/train_groundTruth.npy")
  test_reg = np.load("dataRPN/test_groundTruth.npy")
  print "----> Completed! \n"

  batch_size = len(train_reg)

  theta = np.zeros([batch_size, 1, 3, 3])

  theta[:, 0, 0, 0] = train_reg[:, 3] / float(48)
  theta[:, 0, 1, 1] = train_reg[:, 3] / float(48) 
  theta[:, 0, 0, 2] = (train_reg[:, 1] - 24) / float(24)
  theta[:, 0, 1, 2] = (train_reg[:, 2] - 24) / float(24)
  theta[:, 0, 2, 2] = 1

  imgs = np.zeros([batch_size, 48, 48, 3])
  imgs[:, :, :, 0] = train_imgs[:, 0, :, :]
  imgs[:, :, :, 1] = train_imgs[:, 1, :, :]
  imgs[:, :, :, 2] = train_imgs[:, 2, :, :]

  test_transformer(imgs[2230, :, :, :], theta[2230, 0, :, :])
