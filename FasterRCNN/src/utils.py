'''
  This file includes helper functions for the project such as dataloader and visualization
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

'''
  Convert reg raw data into proper array form
'''
def convertReg(rawData):
  total = len(rawData)
  data = np.zeros([len(rawData), 3, 6, 6])

  ind = np.arange(0, total, 1)
  for i in ind:
    print 'Processing the {}th ground truth......'.format(i)
    cur_data = rawData[i, :]

    data[i, 0, :, :] = np.full((6, 6), cur_data[0])
    data[i, 1, :, :] = np.full((6, 6), cur_data[1])
    data[i, 2, :, :] = np.full((6, 6), cur_data[2])

  np.save('regdata.npy', data)
    


  # pdb.set_trace()


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
  plot curver between iteration times and gradient
  - Input iteration: the iteration times of training
  - Iuput loss: loss value during the training process
  - Input accuracy: prediction accuracy during the training process
'''
def processPlot_grad(iteration, grad_w_front, grad_w_back, title):
  fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))

  x = np.arange(0, iteration, 1)

  Ax0.plot(x, grad_w_front)
  # Ax0.text(0.5, 80, , fontsize=12)
  Ax0.set_title('Gradient of Loss wrt Weight (First Conv Layer)') 
  Ax0.set_xlabel('iteration times')
  Ax0.set_ylabel('Gradient Magnitude')
  Ax0.grid(True)
  

  Ax1.plot(x, grad_w_back)
  # Ax0.text(0.5, 80, , fontsize=12)
  Ax1.set_title('Gradient of Loss wrt Weight (Last Conv Layer)') 
  Ax1.set_xlabel('iteration times')
  Ax1.set_ylabel('Gradient Magnitude')
  Ax1.grid(True)

  plt.suptitle(title, fontsize=16)
  plt.show()


'''
  display a batch size of images: input should be DataLoader type
'''
def cifarImshow(data):
  data_iter = iter(data)
  im_cur, label_cur = data_iter.next()
  # print im_cur.size()
  im_cur = utils.make_grid(im_cur)

  np_img = im_cur.numpy()
  plt.imshow(np.transpose(np_img, (1, 2, 0)))
  plt.show()


'''
  plot accuracy and loss curve wrt the iteration times
'''
def plotCurve(net_num, processType):
  resfile_train = './res/Net{}_train_res_processType_{}.npy'.format(net_num, processType)
  resfile_test = './res/Net{}_test_res_processType_{}.npy'.format(net_num, processType)
  resfile_grad = './res/Net{}_train_res_grad_processType_{}.npy'.format(net_num, processType)

  resTrain = np.load(resfile_train)
  resTest = np.load(resfile_test)
  resGrad = np.load(resfile_grad)

  train_loss, train_acc = resTrain[0, :, :], resTrain[1, :, :]
  grad_w_front, grad_w_back = resGrad[0, :, :], resGrad[1, :, :]
  test_loss, test_acc = resTest[0], resTest[1]

  total = train_loss.shape[0]
  title = 'Training Loss and Accuracy Curves wrt the iteration (Include Test Result)'
  processPlot_acc_loss(total, train_loss, train_acc, test_loss, test_acc, title)

  title = 'Training Gradient Curves wrt the iteration'
  processPlot_grad(total, grad_w_front, grad_w_back, title)


'''
  data loader of cifar 10
'''
def dataloader(forTrain, processType, bs):
  '''
    root: the directory to store data
    train: true means reading in training data, false represent test data loading
    download: false means do not download data form website
    transform: none means using the raw image without any modification
  '''
  trans = obtainTransform(forTrain, processType)
  dataSet = datasets.CIFAR10(root='./data', train=forTrain, download=False, transform=trans)
  # set batch size and shuffle data
  data_loader = torch.utils.data.DataLoader(dataSet, batch_size=bs, shuffle=True)

  return data_loader


'''
  define data transform
'''
def obtainTransform(forTrain, processType):
  if forTrain:
    # raw data
    if processType == 0:
      trans = transforms.Compose([transforms.ToTensor()])

    # normalize raw image
    elif processType == 1:
      trans = transforms.Compose([
        transforms.ToTensor(),   # convert to tensor type for normalization
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # normalize rgb channels
      ])

    # norm + random flip 
    elif processType == 2:
      trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # convert to PIL image first for flip
        transforms.RandomHorizontalFlip(),  # flip images horizontally randomly (50%)
        transforms.ToTensor(),   # convert to tensor type for normalization
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # normalize rgb channels
      ])

    # norm + random flip + pad + random crop
    elif processType == 3:
      trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # convert to PIL image first for flip
        transforms.RandomHorizontalFlip(),  # flip images horizontally randomly (50%)
        transforms.ToTensor(),   # convert to tensor type for normalization
        transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # normalize rgb channels
        transforms.ToPILImage(),
        transforms.Pad(padding=4, fill=(0, 0, 0)),  # pad 4 zeros on each side for all channels
        transforms.RandomCrop(size=(32, 32), padding=0),  # random crop to get 32x32 size of image
        transforms.ToTensor()
      ])

  # no extra pre-processing for test data
  else:
    trans = transforms.Compose([transforms.ToTensor()])

  return trans


'''
  feature map visualization
'''
def featureMapVis(processType, batchSize):
  test_data = dataloader(forTrain=False, processType=processType, bs=batchSize)
  model_path = './models/model_dataProcessType_{}.pth'.format(processType)

  the_model = torch.load(model_path)
  the_model.eval()

  data_iter = iter(test_data)
  x, label_cur = data_iter.next()
  x = Variable(x)

  ori = x.data.numpy()
  ori_im = np.zeros([32, 32, 3])

  ori_im[:, :, 0] = ori[0, 0, :, :]
  ori_im[:, :, 1] = ori[0, 1, :, :]
  ori_im[:, :, 2] = ori[0, 2, :, :]


  # conv1 + bacth norm + relu + avg pool
  x = the_model.conv1(x)

  x = F.relu(the_model.batchNorm1(x))
  x = the_model.avgPool(x)

  # conv2 + bacth norm + relu + avg pool
  x = the_model.conv2(x)
  x = F.relu(the_model.batchNorm2(x))
  x = the_model.avgPool(x)

  # conv1 + bacth norm + relu + avg pool
  x = the_model.conv3(x)
  # x = F.relu(the_model.batchNorm3(x))
  # x = the_model.avgPool(x)


  fm_c1 = x.data.numpy()
  fm_c1_4 = fm_c1[0, 3, :, :]
  fm_c1_12 = fm_c1[0, 11, :, :]
  fm_c1_18 = fm_c1[0, 17, :, :]
  fm_c1_26 = fm_c1[0, 25, :, :]
  fm_c1_30 = fm_c1[0, 29, :, :]

  fig, (Ax0, Ax1, Ax2, Ax3, Ax4, Ax5) = plt.subplots(1, 6, figsize = (8, 8))

  Ax0.set_title('Ori Img') 
  Ax0.imshow(ori_im, cmap='gray', interpolation='nearest')
  Ax0.axis('off')

  Ax1.set_title('C4')
  Ax1.imshow(fm_c1_4)
  Ax1.axis('off')

  Ax2.set_title('C12')
  Ax2.imshow(fm_c1_12)
  Ax2.axis('off')

  Ax3.set_title('C18')
  Ax3.imshow(fm_c1_18)
  Ax3.axis('off')

  Ax4.set_title('C26')
  Ax4.imshow(fm_c1_26)
  Ax4.axis('off')

  Ax5.set_title('C30')
  Ax5.imshow(fm_c1_30)
  Ax5.axis('off')

  plt.show()


# '''
#   convert raw images and mask for traning usage
# '''
# def rawImAndMaskConvert():
#   # folder name that stores images and mask
#   folder_im = 'cifar10_transformed/imgs'
#   folder_mask = 'cifar10_transformed/masks'

#   # file to store training and test ground truth
#   file_train = 'cifar10_transformed/devkit/train.txt'
#   file_test = 'cifar10_transformed/devkit/test.txt'

#   # with open(file_train, "r") as ins:
#   #   train_gt = np.zeros([10000, 4])
#   #   # test_gt = np.zeros([10000, 4])
#   #   ind = 0
#   #   for line in ins:
#   #     img_name, label, c_row, c_col, width = line.split(" ")
#   #     print "The ground truth of " + img_name

#   #     train_gt[ind, 0] = int(label)
#   #     train_gt[ind, 1], train_gt[ind, 2] = int(c_row), int(c_col)
#   #     train_gt[ind, 3] = int(width)

#   #     # test_gt[ind, 0] = int(label)
#   #     # test_gt[ind, 1], test_gt[ind, 2] = int(c_row), int(c_col)
#   #     # test_gt[ind, 3] = int(width)

#   #     ind += 1

#   # np.save('dataRPN/train_groundTruth', train_gt)
#   # np.save('dataRPN/test_groundTruth', test_gt)

#   # read images one by one
#   # imgs_train = np.zeros([10000, 3, 48, 48])
#   # imgs_test = np.zeros([10000, 3, 48, 48])

#   mask_train = np.zeros([10000, 1, 6, 6])
#   mask_test = np.zeros([10000, 1, 6, 6])

#   img_ind = 0
#   for filename in os.listdir(folder_mask):
#     print "convert the {}th image .............".format(img_ind)
#     # read in image and convert color space for better visualization
#     im_path = os.path.join(folder_mask, filename)
#     im_cur = np.array(Image.open(im_path).convert('RGB'))

#     # if img_ind < 10000:
#     #   imgs_train[img_ind, 0, :, :] = im_cur[:, :, 0]
#     #   imgs_train[img_ind, 1, :, :] = im_cur[:, :, 1]
#     #   imgs_train[img_ind, 2, :, :] = im_cur[:, :, 2]

#     # else:
#     #   imgs_test[img_ind - 10000, 0, :, :] = im_cur[:, :, 0]
#     #   imgs_test[img_ind - 10000, 1, :, :] = im_cur[:, :, 1]
#     #   imgs_test[img_ind - 10000, 2, :, :] = im_cur[:, :, 2]

#     if img_ind < 10000:
#       mask_train[img_ind, 0, :, :] = im_cur[:, :, 0]
#       # mask_train[img_ind, 1, :, :] = im_cur[:, :, 1]
#       # mask_train[img_ind, 2, :, :] = im_cur[:, :, 2]

#     else:
#       mask_test[img_ind - 10000, 0, :, :] = im_cur[:, :, 0]
#       # mask_test[img_ind - 10000, 1, :, :] = im_cur[:, :, 1]
#       # mask_test[img_ind - 10000, 2, :, :] = im_cur[:, :, 2]


#     img_ind += 1

#   # np.save('train_imgs.npy', imgs_train)
#   # np.save('test_imgs.npy', imgs_test)
#   np.save('train_mask.npy', mask_train)
#   np.save('test_mask.npy', mask_test)

# if __name__ == "__main__":
#   rawImAndMaskConvert()
