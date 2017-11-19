'''
  This file includes helper functions for the project such as dataloader and visualization
'''
import numpy as np
# import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def str2bool(v):
  return v.lower() in ('true', '1') 

'''
  update learning rate
'''
def adjust_lr(optimizer, lr_decay_factor):
  for param_group in optimizer.param_groups:
    cur_lr = param_group['lr']
    lr = cur_lr * lr_decay_factor 
    param_group['lr'] = lr
  
  # print('===> cur_lr %.5f, updated lr %.5f, decay factor %.4f'%(cur_lr, lr, lr_decay_factor))
  return lr 


'''
  build anchor box
'''
def build_anchor(size=6, scale=8, start=4):
  x, y = np.meshgrid(range(size), range(size))
  x, y = x.reshape(-1), y.reshape(-1)

  inds = np.vstack([y, x])
  inds = start + scale * inds

  w_array = 32 * np.ones(36)

  anchor = np.vstack([inds, w_array])
  return anchor


'''
  Parameterize the coordinates
  - input reg_pred: the reg_map (batch_size x 3 x 36)
  - input anchors: the anchor box (batch_size x 3 x 36) 
  - output normed: same size as the input but normalized form
'''
def paramCoor(raw, anchor):
  # deal with ground truth
  if raw.ndimension() < 3:
    raw = raw.unsqueeze(2)
    raw = raw.expand_as(anchor)

  x = torch.div(raw[:, 0, :] - anchor[:, 0, :], anchor[:, 2, :])
  y = torch.div(raw[:, 1, :] - anchor[:, 1, :], anchor[:, 2, :])
  w = torch.div(raw[:, 2, :], anchor[:, 2, :]).log()

  x = x.unsqueeze(dim=1)
  y = y.unsqueeze(dim=1)
  w = w.unsqueeze(dim=1)

  normed = torch.cat([x, y, w], dim=1)

  return normed

'''
  Smooth l1 loss: the output is batch_size x 36
  - input: delta_t : batch_size x 3 x 36
  - output: l1_loss : batch_size x 36
'''
def smoothL1Loss(delta_t):
  if isinstance(delta_t, torch.Tensor):
    delta_t = delta_t.data 
    norm_delta = torch.abs(delta_t) # N x 3 x d , absolute values 

    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 - torch.mul(one_mask, torch.FloatTensor([0.5])) ) * norm_delta 
    add_factor = torch.mul((1 - one_mask), torch.FloatTensor([-0.5]))

  else:
    norm_delta = torch.abs(delta_t) # N x 3 x d , absolute values 
    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 - 0.5 * one_mask) * norm_delta
    add_factor = -0.5 * (1 - one_mask) 
  
  l1_loss = multi_factor * norm_delta + add_factor
  return l1_loss.sum(dim=1)


'''
  Compute the regression loss
  - input reg_pred: the reg_map (batch_size x 3 x 36)
  - input reg_gt: the box ground truth (batch_size x 3)
  - input mask_pos: the mask to show pos positions (batch_size x 36)
  - input anchors: the anchor box (3 x 36)
'''
def build_reg_loss(reg_pred, reg_gt, mask_pos, anchors):
  bs = reg_pred.size(0)

  # convert the anchor box into size batch_size x 3 x 36
  if anchors.ndimension() < 3:
    anchors.unsqueeze(2)
    anchors = anchors.expand_as(reg_pred)

  # parameterize coordinates
  regPred_norm = paramCoor(reg_pred, anchors)
  regGt_norm = paramCoor(reg_gt, anchors)

  # compute smooth L1 loss
  reg_loss = smoothL1Loss(regPred_norm - regGt_norm)

  reg_loss = (reg_loss * mask_pos).sum() / (mask_pos.sum())
  return reg_loss


'''
  compute the cls loss
'''
def build_cls_loss(cls_pred, mask, valid_mask):
  isobject_criterion = nn.BCEWithLogitsLoss(valid_mask) 
  cls_loss  = isobject_criterion(cls_pred.view(-1), mask.view(-1))
  return cls_loss


'''
  Compute accuracy: cls accuracy and reg accuracy
'''
# build cls accuracy
def build_cls_acc(cls_pred, mask, indp, indn):
  threshold_pos, threshold_neg = 0.75, 0.25
  predPos = torch.ge(cls_pred, threshold_pos).type(torch.FloatTensor).cuda(1)
  predNeg = torch.ge(cls_pred, threshold_neg).type(torch.FloatTensor).cuda(1)

  pos_match = predPos.eq(mask).type(torch.FloatTensor).cuda(1) * indp
  neg_match = predNeg.eq(mask).type(torch.FloatTensor).cuda(1) * indn

  correct = pos_match.sum() + neg_match.sum()
  acc = correct / (indp.sum() + indn.sum())

  return acc.data[0]


# build reg accuracy: use the abs distance error
def build_reg_acc(reg_pred, reg_gt, indp, bs):
  if reg_gt.ndimension() < 3:
    reg_gt = reg_gt.unsqueeze(2)
    reg_gt = reg_gt.expand_as(reg_pred)
  
  diff = torch.abs(reg_pred - reg_gt).sum(dim=1)
  
  # only count the pos positions
  valid_diss = diff * indp
  valid_diss_sum = valid_diss.sum().data[0]

  return valid_diss_sum / float(bs)




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
# def processPlot_grad(iteration, grad_w_front, grad_w_back, title):
#   fig, (Ax0, Ax1) = plt.subplots(1, 2, figsize = (16, 8))

#   x = np.arange(0, iteration, 1)

#   Ax0.plot(x, grad_w_front)
#   # Ax0.text(0.5, 80, , fontsize=12)
#   Ax0.set_title('Gradient of Loss wrt Weight (First Conv Layer)') 
#   Ax0.set_xlabel('iteration times')
#   Ax0.set_ylabel('Gradient Magnitude')
#   Ax0.grid(True)
  

#   Ax1.plot(x, grad_w_back)
#   # Ax0.text(0.5, 80, , fontsize=12)
#   Ax1.set_title('Gradient of Loss wrt Weight (Last Conv Layer)') 
#   Ax1.set_xlabel('iteration times')
#   Ax1.set_ylabel('Gradient Magnitude')
#   Ax1.grid(True)

#   plt.suptitle(title, fontsize=16)
#   plt.show()


'''
  display a batch size of images: input should be DataLoader type
'''
# def cifarImshow(data):
#   data_iter = iter(data)
#   im_cur, label_cur = data_iter.next()
#   # print im_cur.size()
#   im_cur = utils.make_grid(im_cur)

#   np_img = im_cur.numpy()
#   plt.imshow(np.transpose(np_img, (1, 2, 0)))
#   plt.show()


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
# '''
# def featureMapVis(processType, batchSize):
#   test_data = dataloader(forTrain=False, processType=processType, bs=batchSize)
#   model_path = './models/model_dataProcessType_{}.pth'.format(processType)

#   the_model = torch.load(model_path)
#   the_model.eval()

#   data_iter = iter(test_data)
#   x, label_cur = data_iter.next()
#   x = Variable(x)

#   ori = x.data.numpy()
#   ori_im = np.zeros([32, 32, 3])

#   ori_im[:, :, 0] = ori[0, 0, :, :]
#   ori_im[:, :, 1] = ori[0, 1, :, :]
#   ori_im[:, :, 2] = ori[0, 2, :, :]


#   # conv1 + bacth norm + relu + avg pool
#   x = the_model.conv1(x)

#   x = F.relu(the_model.batchNorm1(x))
#   x = the_model.avgPool(x)

#   # conv2 + bacth norm + relu + avg pool
#   x = the_model.conv2(x)
#   x = F.relu(the_model.batchNorm2(x))
#   x = the_model.avgPool(x)

#   # conv1 + bacth norm + relu + avg pool
#   x = the_model.conv3(x)
#   # x = F.relu(the_model.batchNorm3(x))
#   # x = the_model.avgPool(x)


#   fm_c1 = x.data.numpy()
#   fm_c1_4 = fm_c1[0, 3, :, :]
#   fm_c1_12 = fm_c1[0, 11, :, :]
#   fm_c1_18 = fm_c1[0, 17, :, :]
#   fm_c1_26 = fm_c1[0, 25, :, :]
#   fm_c1_30 = fm_c1[0, 29, :, :]

#   fig, (Ax0, Ax1, Ax2, Ax3, Ax4, Ax5) = plt.subplots(1, 6, figsize = (8, 8))

#   Ax0.set_title('Ori Img') 
#   Ax0.imshow(ori_im, cmap='gray', interpolation='nearest')
#   Ax0.axis('off')

#   Ax1.set_title('C4')
#   Ax1.imshow(fm_c1_4)
#   Ax1.axis('off')

#   Ax2.set_title('C12')
#   Ax2.imshow(fm_c1_12)
#   Ax2.axis('off')

#   Ax3.set_title('C18')
#   Ax3.imshow(fm_c1_18)
#   Ax3.axis('off')

#   Ax4.set_title('C26')
#   Ax4.imshow(fm_c1_26)
#   Ax4.axis('off')

#   Ax5.set_title('C30')
#   Ax5.imshow(fm_c1_30)
#   Ax5.axis('off')

#   plt.show()


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