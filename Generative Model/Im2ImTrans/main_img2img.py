from __future__ import print_function
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf

from utils import *
from dataLoader import *
from img2img import img2img

from utils import *
from dataLoader import *
from model import *


slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)


################
# Define flags #
################
flags = tf.app.flags
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir", 'im2im_res/samples', "Directory to save samples")
flags.DEFINE_string("curvedir", 'im2im_res/curves', "Directory to save curves and loss data")
flags.DEFINE_integer("total_num", 88, "The size of the dataset [88]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [1]")
flags.DEFINE_integer("sample_size", 1, "The size of sample images [1]")
flags.DEFINE_integer("image_height", 256, "The height of images [256]")
flags.DEFINE_integer("image_width", 256, "The width of images [256]")
flags.DEFINE_integer("sketch_channel", 1, "The channel number of sketch [1]")
flags.DEFINE_integer("photo_channel", 3, "The channel number of photo [3]")
flags.DEFINE_integer("max_iteration", 2001, "The max iteration times [1e6]")
flags.DEFINE_integer("interval_plot", 1, "The epoch interval to plot training loss [1000]")
flags.DEFINE_integer("interval_save", 10, "The epoch interval to save generative images [1000]")
flags.DEFINE_integer("dim_first_modelG", 64, "The filter number of first conv layer in model G")
flags.DEFINE_integer("dim_first_modelD", 64, "The filter number of first deconv layer in model D")
flags.DEFINE_float("L1_loss_weight", 100, "The relative weight for L1 loss")
flags.DEFINE_float("lr_modelG", 0.0002, "The learning rate for model G")
flags.DEFINE_float("lr_modelD", 0.0002, "The learning rate for model D")
flags.DEFINE_float("beta1", 0.5, "The momentum term of Adam optimizer")
flags.DEFINE_boolean("addNoise", False, "True if adding noise to the raw data")
flags.DEFINE_boolean("debug", False, "True if debug mode")
flags.DEFINE_boolean("curveShow", False, "True if show loss curves")
FLAGS = flags.FLAGS



########
# Main #
########
def main(cufs=True):
  # load data
  skt_dict, pht_dict = dataloader_cufs_students()

  train_skt, train_pht = skt_dict['train'], pht_dict['train']
  test_skt, test_pht = skt_dict['test'], pht_dict['test']

  # obtain output channel number (e.g. 1 for grayscale or 3 for rgb)
  FLAGS.total_num = train_skt.shape[0]
  FLAGS.sketch_channel = train_skt.shape[-1]
  FLAGS.photo_channel = train_pht.shape[-1]

  # check necessary directories
  print ("\n>>>>>>>>>>> Check necessary directories..............")
  if not os.path.exists(FLAGS.sampledir):
    os.makedirs(FLAGS.sampledir)

  if not os.path.exists(FLAGS.curvedir):
    os.makedirs(FLAGS.curvedir)
  

  with tf.Session() as sess:
    # initialize img2img model
    model = img2img(sess, im_h=FLAGS.image_height, im_w=FLAGS.image_width, 
                    im_c_x=FLAGS.sketch_channel, im_c_y=FLAGS.photo_channel,
                    batch_size=FLAGS.batch_size, sample_size=FLAGS.sample_size, max_iteration=FLAGS.max_iteration, 
                    interval_plot=FLAGS.interval_plot, interval_save=FLAGS.interval_save, 
                    curve_dir=FLAGS.curvedir, sampler_dir=FLAGS.sampledir, 
                    G_f_dim=FLAGS.dim_first_modelG, D_f_dim=FLAGS.dim_first_modelD, L1_lambda=FLAGS.L1_loss_weight, 
                    train_sketches=train_skt, train_photos=train_pht, test_sketches=test_skt, test_photos=test_pht)

    # show curves
    if FLAGS.curveShow:
      processPlot_GANs()

    else:
      # train model
      model.train(FLAGS)



if __name__ == '__main__':
  main()


