import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
  sigmoid function
  - Input w: value that represents weight
  - Input b: value that represents bias
  - Input x: a list of variables (ndarray type)
  - Output y: a list of sigmoid value with same size with x
'''
def sigmoid(w, b, x):
  temp = w * x + b
  return 1 / (1 + np.exp(-temp))


'''
  L2 loss computation
  - Input y_GT: the variable represents the ground truth
  - Input y_Pred: the variable represents the prediction value
  - Output loss: L2 loss
'''
def lossL2(y_Pred, y_GT):
  diff2 = np.square(y_Pred - y_GT)
  return diff2

'''
  cross-entropy loss computation
  - Input y_GT: the variable represents the ground truth
  - Input y_Pred: the variable represents the prediction value
  - Output loss: cross-entropy loss
'''
def lossCrossEntropy(y_Pred, y_GT):
  diff_cross = y_GT * np.log(y_Pred) + (1 - y_GT) * np.log(1 - y_Pred)
  return -diff_cross


'''
  Gradient of the L2 loss wrt the weight
  - Input y_GT: the variable represents the ground truth
  - Input W: the variable represents the weight
  - Input b: the variable represents the bias
  - Input x: the variable represents the feature
  - Output grad: the gradient of loss wrt the weight
'''
def gradL2Loss2W(y_GT, W, b, x):
  y_Pred = sigmoid(W, b, x)

  grad_Loss2yPred = 2 * (y_Pred - y_GT)
  grad_yPred2W = x * np.square(y_Pred) * np.exp(-1 * (W * x + b))

  grad_Loss2W = grad_Loss2yPred * grad_yPred2W

  # the Simplification form
  # grad_Loss2W = -2 * x * (y_GT - y_Pred) * (1 - y_Pred) * y_Pred

  return grad_Loss2W


'''
  Gradient of the Cross Entropy loss wrt the weight
  - Input y_GT: the variable represents the ground truth
  - Input W: the variable represents the weight
  - Input b: the variable represents the bias
  - Input x: the variable represents the feature
  - Output grad: the gradient of Cross-Entropy loss wrt the weight
'''
def gradCELoss2W(y_GT, W, b, x):
  y_Pred = sigmoid(W, b, x)

  grad_Loss2yPred = ((1 - y_GT) / (1 - y_Pred)) - (y_GT / y_Pred)
  grad_yPred2W = y_Pred * x * (1 - y_Pred)

  grad_Loss2W = grad_Loss2yPred * grad_yPred2W

  # # # the Simplification form
  # grad_Loss2W = x * (y_Pred - y_GT)

  return grad_Loss2W



'''
  Customized Activation function: leakyReLU
'''
def leaky_relu(W, b, x):
  y = W * x + b
  alpha = 0.1
  # positive part should have same result as relu(x)
  y[y <= 0] *= alpha
  return y


'''
  Customized Activation function: ELU(Exponential Linear Unit)
'''
def ELU(W, b, x):
  y = W * x + b

  alpha = 2
  y[y <= 0] = alpha * (np.exp(y[y <= 0]) - 1)
  return y


'''
  Loss gradient wrt weight
'''
def ELUlossGrad2Weight(y_GT, W, b, x):
  val = W * x + b
  y_Pred = ELU(W, b, x)

  grad_Loss2W = 2 * (y_Pred - y_GT)
  
  alpha = 2
  grad_Loss2W[val > 0] *= x
  grad_Loss2W[val <= 0] *= alpha * x * np.exp(val[val <= 0])

  return grad_Loss2W


'''
  Loss gradient wrt bias
'''
def ELUlossGrad2bias(y_GT, W, b, x):
  val = W * x + b
  y_Pred = ELU(W, b, x)

  grad_Loss2b = 2 * (y_Pred - y_GT)
  
  alpha = 2
  grad_Loss2b[val > 0] *= 1
  grad_Loss2b[val <= 0] *= alpha * 1 * np.exp(val[val <= 0])

  return grad_Loss2b

