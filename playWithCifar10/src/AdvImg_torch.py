import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb
import numpy as np
import matplotlib.pyplot as plt

from utils import * 

'''
  Different file contains various models
'''
from p1_model_torch import *


# define global variables
train_display_interval = 10
batch_size = 100
step = 100
max_iteration = 50

label = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 
         4 : 'deer', 5 : 'dog', 6 : 'frog', 7 : 'horse', 8 : 'ship', 9 : 'truck'}

'''
  training model
  - input dataload of training images and corresponding label
'''
def geneAdver(model, trainData, trueGT_true, trainGT_wrong):
  optimizer = optim.SGD(model.parameters(), lr = 0.1)

  perturbation = Variable(torch.from_numpy(np.zeros([1, 3, 32, 32])))
  perturbation = perturbation.type(torch.FloatTensor)

  while True:
    # training process
    optimizer.zero_grad()
    label_pred = model(trainData)  # the output is log softmax value
    
    # compute confidence
    label_pred_np = label_pred.data.numpy()
    exp_pred = np.exp(label_pred_np)
    confidence = exp_pred[0][trainGT_wrong.data.numpy()[0]] / exp_pred.sum()

    print 'The current confidence of wrong label {} is {:.8f}% (The ground label is {}).'.format(
      trainGT_wrong.data.numpy()[0], confidence * 100, trueGT_true)

    if confidence >= 0.99:
      break

    # compute the accuracy
    pred = label_pred.data.max(1, keepdim = True)[1] # get the index of the max log-probability

    # nll loss is suitable for n classification problem
    loss = F.nll_loss(label_pred, trainGT_wrong)
    loss.backward()

    img_grad = trainData.grad.clone()
    perturbation.data -= 0.0007 * torch.sign(img_grad.data)
    # clean up gradient
    trainData.grad.data.zero_()

    trainData.data -= 0.0007 * torch.sign(img_grad.data)

  return trainData, perturbation


'''
  obtain an instance of each object class
'''
def obtainImage(dataLoader):
  atch_idx = 0
  data_iter = iter(dataLoader)
  batch_cur, label_cur = data_iter.next()
  batch_np, label_np = batch_cur.numpy(), label_cur.numpy()

  cls_ind = 0
  obj_ind = np.zeros([10])
  while cls_ind < 10:
    ind_set = np.where(label_np == cls_ind)
    obj_ind[cls_ind] = ind_set[0][0]
    cls_ind += 1

  obj_ind = obj_ind.astype(int)

  img_set = batch_np[obj_ind]

  return img_set


'''
  display function: the input should be Variable
'''
def dispIm(data_ori, perturbation, data_adv, label_gt, label_adv):
  # convert variable to image
  im_ori = np.zeros([32, 32, 3])
  im_ori[:, :, 0] = data_ori.data.numpy()[0, 0, :, :]
  im_ori[:, :, 1] = data_ori.data.numpy()[0, 1, :, :]
  im_ori[:, :, 2] = data_ori.data.numpy()[0, 2, :, :]

  # convert variable to image
  im_pert = np.zeros([32, 32, 3])
  im_pert[:, :, 0] = perturbation.data.numpy()[0, 0, :, :]
  im_pert[:, :, 1] = perturbation.data.numpy()[0, 1, :, :]
  im_pert[:, :, 2] = perturbation.data.numpy()[0, 2, :, :]

  # convert variable to image
  im_adv = np.zeros([32, 32, 3])
  im_adv[:, :, 0] = data_adv.data.numpy()[0, 0, :, :]
  im_adv[:, :, 1] = data_adv.data.numpy()[0, 1, :, :]
  im_adv[:, :, 2] = data_adv.data.numpy()[0, 2, :, :]

  fig, (Ax0, Ax1, Ax2) = plt.subplots(1, 3, figsize = (8, 8))

  Ax0.set_title('Original Img(' + label[label_gt] + ')') 
  Ax0.imshow(im_ori, cmap='gray', interpolation='nearest')
  Ax0.axis('off')

  Ax1.set_title('Perturbation') 
  Ax1.imshow(im_pert, cmap='gray', interpolation='nearest')
  Ax1.axis('off')
  
  Ax2.set_title('Adversarial Img(' + label[label_adv] + ')')
  Ax2.imshow(im_adv, cmap='gray', interpolation='nearest')
  Ax2.axis('off')

  plt.show()


'''
  main part: construct model, training and testing
'''
def main(processType):
  # load data
  train_data = dataloader(forTrain=True, processType=processType, bs=batch_size)
  img_set = obtainImage(train_data)

  # load model
  model_path = './models/model_dataProcessType_{}.pth'.format(processType)
  the_model = torch.load(model_path)
  the_model.eval()

  optimizer = optim.SGD(the_model.parameters(), lr = 0.1)

  adv_imgs, adv_label = np.zeros([10, 3, 32, 32]), np.zeros([10])

  i = 0
  while i < 10:
    data = torch.from_numpy(np.zeros([1, 3, 32, 32]))
    disp = torch.from_numpy(np.zeros([1, 3, 32, 32]))  # temp torch for visualization

    data[0, :, :, :] = torch.from_numpy(img_set[i, :, :, :])
    disp[0, :, :, :] = torch.from_numpy(img_set[i, :, :, :])

    label_cur = torch.from_numpy(np.asarray([(i + 1) % 10]))

    data = data.type(torch.FloatTensor)
    disp = disp.type(torch.FloatTensor)

    label_cur = label_cur.type(torch.LongTensor)

    data, label_cur = Variable(data, requires_grad=True), Variable(label_cur)
    disp = Variable(disp)

    # generate Adversarial image
    adv, perb = geneAdver(the_model, data, i, label_cur)
    
    adv_imgs[i, :, :, :] = adv.data.numpy()[0, :, :, :]
    adv_label[i] = i

    # display the adversarial image
    # dispIm(disp, perb, adv, i, (i + 1) % 10)

    i += 1

  # load fake images
  img_path_1, im_gt_1 = './res_fig/p3/advImgs_1.npy', './res_fig/p3/advImgs_gt_1.npy'
  # img_path_2, im_gt_2 = './res_fig/p3/advImgs_2.npy', './res_fig/p3/advImgs_gt_2.npy'

  np.save(img_path_1, adv_imgs)
  np.save(im_gt_1, adv_label)


if __name__ == '__main__':

  '''
    processType represents different data pre-process type
    0: raw data
    1: normalize the raw data
    2: normalize + random flip raw data
    3: normalize + random flip + pad + crop raw data
  '''
  processType = 2
  main(processType)

  '''
    plot train loss and acc curve as well as test result
  '''
  # plotCurve(processType)
  
  '''
    plot feature map in different ConvNet layers
  '''
  # featureMapVis(processType, batch_size)


