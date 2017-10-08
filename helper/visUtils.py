from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import p1 as F

'''
  Surface plot function for sigmoid
'''
def sigmoidPlot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-5, 5, 0.25)
  b = np.arange(-5, 5, 0.25)

  W, b = np.meshgrid(W, b)

  sig = F.sigmoid(W, b, 1)

  # Plot the surface.
  surf = ax.plot_surface(W, b, sig, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  ax.plot([0], [0], [F.sigmoid(0, 0, 1)], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

# Putting the limits in the axes
  ax.set_title("Sigmoid Function")
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("sigmoid")

  # Customize the z axis.
  ax.set_zlim(0.0, 1.2)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


'''
  Surface plot function for leaky ReLU
'''
def CustomizedActivationPlot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-5, 5, 0.25)
  b = np.arange(-5, 5, 0.25)

  W, b = np.meshgrid(W, b)

  sig = F.ELU(W, b, 1)
  # sig = F.leaky_relu(W, b, 1)

  # Plot the surface.
  surf = ax.plot_surface(W, b, sig, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)

# Putting the limits in the axes
  ax.set_title('ELU Function with alpha {}'.format(2))
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("ELU")

  # Customize the z axis.
  ax.set_zlim(-3, 10)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


'''
  Surface plot function for L2 loss
'''
def lossL2Plot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-5, 5, 0.25)
  b = np.arange(-5, 5, 0.25)

  W, b = np.meshgrid(W, b)

  y_Pred = F.sigmoid(W, b, 1)
  # y_Pred = F.ELU(W, b, 1)

  loss = F.lossL2(y_Pred, 0.5)

  # Plot the surface.
  surf = ax.plot_surface(W, b, loss, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  # ax.plot([0], [0], [F.lossL2(F.sigmoid(0, 0, 1), 0.5)], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

  # Putting the limits in the axes
  ax.set_title("Sigmoid + L2 Loss")
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("loss")

  # Customize the z axis.
  ax.set_zlim(0.0, 1.2)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


'''
  Surface plot function for corss-entropy loss
'''
def lossCrossEntropyPlot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-8, 8, 0.25)
  b = np.arange(-8, 8, 0.25)

  W, b = np.meshgrid(W, b)

  y_Pred = F.sigmoid(W, b, 1)

  loss = F.lossCrossEntropy(y_Pred, 0.5)

  # Plot the surface.
  surf = ax.plot_surface(W, b, loss, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  # ax.plot([0], [0], [F.lossCrossEntropy(F.sigmoid(0, 0, 1), 0.5)], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

  # Putting the limits in the axes
  ax.set_title("Sigmoid + Cross-Entropy Loss")
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("loss")

  # Customize the z axis.
  ax.set_zlim(0.0, 10)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


'''
  Surface plot function for L2 loss gradient wrt weight
'''
def gradL2Loss2WPlot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-5, 5, 0.25)
  b = np.arange(-5, 5, 0.25)

  W, b = np.meshgrid(W, b)

  grad = F.gradL2Loss2W(0.5, W, b, 1)
  # grad = F.ELUlossGrad2bias(0.5, W, b, 1)  # the function to compute gradient of ELU activation
  # grad = F.ELUlossGrad2Weight(0.5, W, b, 1)  # the function to compute gradient of ELU activation

  # Plot the surface.
  surf = ax.plot_surface(W, b, grad, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  # ax.plot([0], [0], [F.gradL2Loss2W(0.5, 0, 0, 1)], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

  # Putting the limits in the axes
  ax.set_title("Gradient of L2 loss wrt weight (Sigmoid Activation)")
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("Gradient")

  # Customize the z axis.
  ax.set_zlim(-5, 20)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)

'''
  Surface plot function for Cross Entropy loss gradient wrt weight
'''
def gradCELoss2WPlot():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  W = np.arange(-5, 5, 0.25)
  b = np.arange(-5, 5, 0.25)

  W, b = np.meshgrid(W, b)

  grad = F.gradCELoss2W(0.5, W, b, 1)

  # Plot the surface.
  surf = ax.plot_surface(W, b, grad, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)
  ax.plot([0], [0], [F.gradCELoss2W(0.5, 0, 0, 1)], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)

  # Putting the limits in the axes
  ax.set_title("Gradient of Cross Entropy loss wrt Weight (Sigmoid Activation)")
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  ax.set_zlabel("Gradient")

  # Customize the z axis.
  ax.set_zlim(-0.8, 0.8)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)


if __name__ == "__main__":

  '''
    uncomment anyone that you want to check
  '''

  # sigmoidPlot()  # plot sigmoid function
  # lossL2Plot()  # plot L2 loss 
  # gradL2Loss2WPlot()  # plot L2 loss gradient

  # lossCrossEntropyPlot()  # plot Cross Entropy loss
  # gradCELoss2WPlot()  # plot Cross Entropy loss gradient

  # CustomizedActivationPlot()  # plot custmoized activation
  
  plt.show()