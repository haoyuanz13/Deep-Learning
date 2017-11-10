import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import pdb


'''
  The Region Proposal Network(RPN)
  - includes the base conv layers to extract feature map and 
  - additional proposal layers for classification and region regression
'''
class RPN(nn.Module):

  def __init__(self):
    super(RPN, self).__init__()
    # Conv blocks: conv->BN->ReLU->max pooling
    def ConvBlockWithMaxPool(c_in, c_out, ks, strs, pad):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=strs, padding=pad),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
      )

    # Conv blocks: conv->BN->ReLU
    def ConvBlockNoMaxPool(c_in, c_out, ks, strs, pad):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=strs, padding=pad),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
      )

    # base net to extract feature map
    self.BaseNet = nn.Sequential(
      ConvBlockWithMaxPool(3, 32, (5, 5), (1, 1), 2),
      ConvBlockWithMaxPool(32, 64, (5, 5), (1, 1), 2),
      ConvBlockWithMaxPool(64, 128, (5, 5), (1, 1), 2),
      ConvBlockNoMaxPool(128, 256, (3, 3), (1, 1), 1)
    )

    # proposal classification branch
    self.prop_cls = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    # proposal regression branch
    # the 1st channel represents the row pos of bbox center
    # the 2nd channel represents the col pos of bbox center
    # the 3rd channel represents the width of bbox
    self.prop_reg = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)

    # loss
    self.cls_loss = None
    self.reg_loss = None

    # ground truth including cls and reg
    self.mask = None
    self.region_gt = None
    self.anchor_scales = None


  # feed forward the data to the model
  def forward(self, im_batch, mask_batch, reg_batch):
    batch_size = len(im_batch)
    # build anchor_scales
    self.anchor_scales = self.anchorBuild(batch_size)

    # convert data into variable
    im_batch = torch.from_numpy(im_batch).type(torch.FloatTensor)
    mask_batch = torch.from_numpy(mask_batch).type(torch.FloatTensor)
    reg_batch = torch.from_numpy(reg_batch).type(torch.FloatTensor)

    # get valid index (pos and neg)
    ind_pos, ind_neg = torch.eq(mask_batch, 1), torch.eq(mask_batch, 0)
    valid_mask = torch.ne(mask_batch, 2).type(torch.FloatTensor)
    validnum = float(ind_pos.sum() + ind_neg.sum())

    # convert tensor to Variable in the cuda type
    im_batch, mask_batch = Variable(im_batch).cuda(1), Variable(mask_batch).cuda(1)
    reg_batch = Variable(reg_batch).cuda(1)
    valid_mask = Variable(valid_mask).cuda(1)

    # store ground truth
    self.mask = mask_batch
    self.region_gt = reg_batch

    # extract feature map
    feaMap = self.BaseNet(im_batch)

    # cls map for proposal classification
    cls_map = F.sigmoid(self.prop_cls(feaMap))

    self.cls_loss = self.build_cls_loss(cls_map, valid_mask, validnum)
    cls_acc = self.build_cls_acc(cls_map, ind_pos, ind_neg, validnum)

    # reg map for proposal regression
    reg_map = self.prop_reg(feaMap)

    self.reg_loss = self.build_reg_loss(reg_map, ind_pos, batch_size)
    reg_acc = self.build_reg_acc(reg_map, ind_pos, batch_size)

    return cls_acc, reg_acc

  '''
    Obtain Loss: cls loss, reg loss and combined loss
  '''
  # the combination loss
  def combLoss(self):
    return self.cls_loss + self.reg_loss

  # the cls loss
  def clsLoss(self):
    return self.cls_loss

  # the reg loss
  def regLoss(self):
    return self.reg_loss


  '''
    Build loss: cls loss and reg loss
  '''
  # build cls loss
  def build_cls_loss(self, cls_pred, vmask, vnum):
    loss_feaMap, loss_mask = vmask * cls_pred, vmask * self.mask
    loss = loss_feaMap * loss_mask + (1 - loss_feaMap) * (1 - loss_mask)
    loss = -loss.log().sum() / vnum

    return loss

  # build reg loss
  def build_reg_loss(self, reg_pred, indp, bs):
    regPred, regGT, anchor_scales = reg_pred.clone(), self.region_gt.clone(), self.anchor_scales.clone()

    # parameterize coordinates
    # regPred[:, 0, :, :] = (regPred[:, 0, :, :] - anchor_scales[:, 0, :, :]) / anchor_scales[:, 2, :, :]
    # regGT[:, 0, :, :] = (regGT[:, 0, :, :] - anchor_scales[:, 0, :, :]) / anchor_scales[:, 2, :, :]

    # regPred[:, 1, :, :] = (regPred[:, 1, :, :] - anchor_scales[:, 1, :, :]) / anchor_scales[:, 2, :, :]
    # regGT[:, 1, :, :] = (regGT[:, 1, :, :] - anchor_scales[:, 1, :, :]) / anchor_scales[:, 2, :, :]

    # regPred[:, 2, :, :] = (regPred[:, 2, :, :] / anchor_scales[:, 2, :, :]).log()
    # regGT[:, 2, :, :] = (regGT[:, 2, :, :] / anchor_scales[:, 2, :, :]).log()

    tx = (regPred[:, 0, :, :] - anchor_scales[:, 0, :, :]) / anchor_scales[:, 2, :, :]
    txs = (regGT[:, 0, :, :] - anchor_scales[:, 0, :, :]) / anchor_scales[:, 2, :, :]

    ty = (regPred[:, 1, :, :] - anchor_scales[:, 1, :, :]) / anchor_scales[:, 2, :, :]
    tys = (regGT[:, 1, :, :] - anchor_scales[:, 1, :, :]) / anchor_scales[:, 2, :, :]

    tw = (regPred[:, 2, :, :] / anchor_scales[:, 2, :, :]).log()
    tws = (regGT[:, 2, :, :] / anchor_scales[:, 2, :, :]).log()

    t, ts = torch.stack([tx, ty, tw], 1), torch.stack([txs, tys, tws], 1)

    # count valid number
    pos_num = float(indp.sum())

    # only count the pos positions
    indp_var = Variable(indp.type(torch.FloatTensor)).cuda(1)

    # own designed smooth l1 loss
    diff = torch.abs(t - ts)
    ind_1, ind_2 = diff.lt(1).type(torch.FloatTensor).cuda(1), diff.ge(1).type(torch.FloatTensor).cuda(1)

    loss1 = (torch.pow(diff, 2) * 0.5) * ind_1
    loss2 = (diff - 0.5) * ind_2

    loss = loss1 * indp_var + loss2 * indp_var

    # regPred_v, regGT_v = t * indp_var, ts * indp_var

    # compute smooth l1 loss
    # loss = F.smooth_l1_loss(regPred_v, regGT_v, size_average=False) / pos_num
    return loss.sum() / pos_num


  '''
    Compute accuracy: cls accuracy and reg accuracy
  '''
  # build cls accuracy
  def build_cls_acc(self, cls_pred, indp, indn, vnum):
    threshold_pos, threshold_neg = 0.75, 0.25
    predPos = torch.ge(cls_pred, threshold_pos).type(torch.FloatTensor).cuda(1)
    predNeg = torch.ge(cls_pred, threshold_neg).type(torch.FloatTensor).cuda(1)

    pos_match = predPos.eq(self.mask).data * indp.cuda(1)
    neg_match = predNeg.eq(self.mask).data * indn.cuda(1)

    correct = pos_match.sum() + neg_match.sum()
    acc = correct / vnum

    return acc

  # build reg accuracy
  def build_reg_acc(self, reg_pred, indp, bs):
    diff = torch.abs(reg_pred - self.region_gt)
    # only count the pos positions
    indp_var = Variable(indp.type(torch.FloatTensor)).cuda(1)

    valid_diss = diff * indp_var
    valid_diss_sum = valid_diss.sum().data[0]

    # pdb.set_trace()
    # build lower and upper boundary for ground truth
    # reg_lowb, reg_upb = self.region_gt.clone(), self.region_gt.clone()

    # threshold_center, threshold_width = 0.5, 1

    # reg_lowb[:, 0:2, :, :] = reg_lowb[:, 0:2, :, :] - threshold_center
    # reg_lowb[:, 2, :, :] = reg_lowb[:, 2, :, :] - threshold_width

    # reg_upb[:, 0:2, :, :] = reg_upb[:, 0:2, :, :] + threshold_center
    # reg_upb[:, 2, :, :] = reg_upb[:, 2, :, :] + threshold_width

    # count valid number
    # pos_num = float(indp.sum())

    # # only count the pos positions
    # indp_var = Variable(indp).cuda(1)

    # ind_low = torch.ge(reg_pred, reg_lowb) * indp_var
    # ind_upp = torch.le(reg_pred, reg_upb) * indp_var

    # ind_cross = ind_low * ind_upp
    # acc = ind_cross.sum().data[0] / pos_num

    return valid_diss_sum / bs


  '''
    Other helper functions
  '''
  # initialize bias
  def parameterSet(self):
    # initialize weights and bias
    self.prop_reg.bias.data[0] = 24
    self.prop_reg.bias.data[1] = 24
    self.prop_reg.bias.data[2] = 32


  # build anchor_scales
  def anchorBuild(self, bs):
    anchor_scales = torch.FloatTensor(bs, 3, 6, 6).zero_()

    anchor_scales[:, 0, :, :] = torch.FloatTensor([[4], [12], [20], [28], [36], [44]]).expand(bs, 6, 6)
    anchor_scales[:, 1, :, :] = torch.FloatTensor([[4], [12], [20], [28], [36], [44]]).expand(bs, 6, 6)
    anchor_scales[:, 2, :, :] = torch.FloatTensor([[32]]).expand(bs, 6, 6)

    anchor_scales = Variable(anchor_scales).cuda(1)

    return anchor_scales







