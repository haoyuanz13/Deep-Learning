from __future__ import print_function
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
import argparse

from utils import *
from dataLoader import *
from model import *
from img2img import img2img
from img2img_x import img2img_x


slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)


################
# Define args  #
################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='cufs_std_concat', help='name of the dataset [cufs_std_concat, cufs_students, facades]')
parser.add_argument('--res_dir', dest='res_dir', default='./im2im_res', help='all results are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='im2im_res/checkpoint', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='im2im_res/test', help='test sample are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='im2im_res/samples', help='samples are saved here')
parser.add_argument('--curve_dir', dest='curve_dir', default='im2im_res/curves', help='loss curves are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='im2im_res/logs', help='graphs are saved here')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='max # images used to train')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--sample_size', dest='sample_size', type=int, default=1, help='# images in sample')
parser.add_argument('--image_height', dest='image_height', type=int, default=256, help='standard image height')
parser.add_argument('--image_width', dest='image_width', type=int, default=256, help='standard image width')
parser.add_argument('--sketch_channel', dest='sketch_channel', type=int, default=3, help='# of input sketch channels')
parser.add_argument('--photo_channel', dest='photo_channel', type=int, default=3, help='# of output photo channels')
parser.add_argument('--interval_plot', dest='interval_plot', type=int, default=1, help='# of epoch intervals to plot training loss')
parser.add_argument('--interval_sample', dest='interval_sample', type=int, default=1, help='# of epoch intervals to sample and save')
parser.add_argument('--interval_save', dest='interval_save', type=int, default=50, help='# of epoch intervals to save model')
parser.add_argument('--dim_first_modelG', dest='dim_first_modelG', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--dim_first_modelD', dest='dim_first_modelD', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--L1_loss_weight', dest='L1_loss_weight', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--lr_modelG', dest='lr_modelG', type=float, default=0.0002, help='initial learning rate for adam [model G]')
parser.add_argument('--lr_modelD', dest='lr_modelD', type=float, default=0.0002, help='initial learning rate for adam [model D]')
parser.add_argument('--n_critic_G', dest='n_critic_G', type=int, default=2, help='# of model G training in each step')
parser.add_argument('--n_critic_D', dest='n_critic_D', type=int, default=1, help='# of model D training in each step')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=1000, help='# of epoch')
parser.add_argument('--addNoise', dest='addNoise', type=bool, default=False, help='if add noise to input data')
parser.add_argument('--debug', dest='debug', type=bool, default=False, help='if debug the model instead of training')
parser.add_argument('--curveShow', dest='curveShow', type=bool, default=False, help='if show loss curves')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--randomSample', dest='randomSample', default=True, help='if sample randomly')
parser.add_argument('--concatSamples', dest='concatSamples', default=True, help='if generate concatenated samples')



# parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
# parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
# parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
# parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
# parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
# parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
# parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
# parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
# parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')

args = parser.parse_args()


########
# Main #
########
def main():
  # check necessary directories
  print ("\n=====>> Checking necessary directories ..........")
  if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

  if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)

  if not os.path.exists(args.curve_dir):
    os.makedirs(args.curve_dir)

  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
  
  if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)
  print ("[*] All necessary directories are EXISTED! \n")


  # build model
  print ("=====>> Initializing img2img model ..........")
  with tf.Session() as sess:
    # read .npy dataset; use slim structure
    if args.dataset_name == "cufs_students":
      model = img2img(sess, im_h=args.image_height, im_w=args.image_width, 
                    im_c_x=args.sketch_channel, im_c_y=args.photo_channel,
                    batch_size=args.batch_size, sample_size=args.sample_size, max_iteration=args.max_iteration, 
                    G_f_dim=args.dim_first_modelG, D_f_dim=args.dim_first_modelD, L1_lambda=args.L1_loss_weight)

    # read the cincatenated data (cufs or facades); include BN layers into the class
    else:
      if args.dataset_name == 'cufs_std_concat':
        print ("\n========== CUFS Studenst Dataset (Raw images) ==========")
      else:
        print ("\n========== Facades Dataset (Raw images) ==========")
        args.max_iteration = 200

      model = img2img_x(sess, im_height=args.image_height, im_width=args.image_width, batch_size=args.batch_size, sample_size=args.sample_size, 
                      output_size=args.image_height, gf_dim=args.dim_first_modelG, df_dim=args.dim_first_modelD, 
                      L1_lambda=args.L1_loss_weight, input_c_dim=args.sketch_channel, output_c_dim=args.photo_channel, 
                      dataset_name=args.dataset_name, checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
    

    # show curves
    if args.curveShow:
      print ("=====>> Showing loss curves ..........")
      processPlot_GANs()

    elif args.phase == 'train':
      # train model
      print ("=====>> [TRAINING] the model ..........")
      model.train(args)

    elif args.phase == 'test':
      print ("=====>> [TESTING] the model ..........")
      model.test(args)


if __name__ == '__main__':
  main()


