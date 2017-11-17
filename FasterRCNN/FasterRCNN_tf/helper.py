'''
  This file includes helper functions for the project such as dataloader and visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb

import tensorflow as tf


'''
  plot curver between iteration times and loss or accuracy
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_acc_loss(iteration, loss, accuracy, test_loss, test_acc, title):
  fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, loss)
  # Ax0.text(0.5, 80, , fontsize=12)
  Ax0.text(0.95, 0.01, 'The average test loss is {:.4f}'.format(test_loss),
        verticalalignment='bottom', horizontalalignment='right', transform=Ax0.transAxes,
        color='red', fontsize=15)

  Ax0.set_title('Loss Value') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('loss')
  Ax0.grid(True)
  

  Ax1.plot(x, accuracy)

  Ax1.text(0.95, 0.01, 'The average test accuracy is {:.4f}%'.format(test_acc * 100),
        verticalalignment='bottom', horizontalalignment='right', transform=Ax1.transAxes,
        color='red', fontsize=15)
  Ax1.set_title('Prediction Accuracy')
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('Accuracy')
  Ax1.grid(True)

  plt.suptitle(title, fontsize=16)
  plt.show()


'''
  plot accuracy and loss curve wrt the iteration times
'''
def processPlot():
  test_res_rcnn = np.load('res/FasterRCNN_test_rcnn.npy') 
  test_res_cls = np.load('res/FasterRCNN_test_cls.npy')
  test_res_reg = np.load('res/FasterRCNN_test_reg.npy')

  train_rcnn_loss = np.load('res/FasterRCNN_train_rcnn_loss.npy')
  train_rcnn_acc = np.load('res/FasterRCNN_train_rcnn_acc.npy')
  train_cls_loss = np.load('res/FasterRCNN_train_cls_loss.npy')
  train_cls_acc = np.load('res/FasterRCNN_train_cls_acc.npy')
  train_reg_loss = np.load('res/FasterRCNN_train_reg_loss.npy')
  train_reg_acc = np.load('res/FasterRCNN_train_reg_acc.npy')

  total = train_cls_loss.shape[0]
  title = 'Cls Branch: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_cls_loss, train_cls_acc, test_res_cls[0], test_res_cls[1], title)

  title = 'Reg Branch: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_reg_loss, train_reg_acc, test_res_reg[0], test_res_reg[1], title)

  title = 'Obj Classification: Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_rcnn_loss, train_rcnn_acc, test_res_rcnn[0], test_res_rcnn[1], title)

