import numpy as np
from IPython.display import IFrame
from IPython.display import display 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os, pdb
import h5py
from utils import TEMPLATE_POINTS, TEMPLATE_VG, plot_voxelgrid, plot_points, array_to_color, disp_image
from voxelgrid import VoxelGrid
from camera_model import CamModel  
from utils import progress_bar

########################################################################
# Changable parameters 
# Choose start index and end index of data points that we want to compute. 
# This help do data processing in parallel 

start_indx = 2600
end_indx = 2700

# Choose mode 'train' or 'test'
mode = 'train'
resume = False 
visual = True 

# Choose directory where raw data are stored 
raw_dir = "data/3d-mnist"

# Choose target directory 
data_dir = raw_dir 

# Choose how fine the voxel discretization is 
N = 60 


########################################################################
# Fixed parameters 
# List of camera poses 
list_poses = [[0,  0, 0], [0, 0, 30], [0, 0, 60], [0, 0, 90], 
              [90, 0, 0], 
              [0, 90, 0]] # [Rx, Ry, Rz]

 
# Source directories 
hfile_dir = os.path.join(raw_dir, mode + '_point_clouds.h5')

# Target directories 
data_dir = raw_dir 
origin_two_D_dir = os.path.join(data_dir, '2D/' + 'orig/') # Store the original 2D image

list_two_D_dirs = []
list_three_D_dirs = []
list_filenames_file = []

for cam_pose_id in range(len(list_poses)):
    # Choose one camera pose ID 
    print('\n===> Camera %d'%cam_pose_id)
    two_D_dir = os.path.join(data_dir, '2D/' + 'pose_' + str(cam_pose_id)) # + '/')
    three_D_dir = os.path.join(data_dir, '3D/' + 'pose_' + str(cam_pose_id)) #) + '/')
    # We may want to run many processes to do data process in parallel. To make the better concurency, we write on different files. However, writing 
    # image data is just fine so we keep do not create replica of directories. 
    filenames_file = os.path.join(data_dir, mode + '_' + 'pose_' + str(cam_pose_id) + '_upto_' + str(end_indx) + '.txt') # i.e. data_dir/train_pose_1_upto_5000

    # Create folders if not existed and/or delete if existed and not resume 
    if not resume: 
        try:
            print('===> Recreate/ Frist time create...Deleting if existed...in %s'%two_D_dir)
            os.remove(filenames_file)
            shutil.rmtree(two_D_dir, ignore_errors=True)
            shutil.rmtree(three_D_dir, ignore_errors=True)
            shutil.rmtree(origin_two_D_dir, ignore_errors=True)

        except:
            print('===> Try to delete %s but fail!'%two_D_dir)
            pass 

    if not os.path.exists(two_D_dir):
        print('===> Dir %s not exist yet. Creating..'%two_D_dir)
        try:
            os.makedirs(two_D_dir)
        except:
            pass 
        try:
            os.makedirs(three_D_dir)
        except:
            pass 
        try: 
            os.makedirs(origin_two_D_dir)
        except:
            pass 
    list_two_D_dirs.append(two_D_dir)
    list_three_D_dirs.append(three_D_dir)
    list_filenames_file.append(filenames_file)

# Open filenames_file for reading 
list_filenames_fopen = [] 
for cam_pose_id in range(len(list_poses)):
    filenames_file = list_filenames_file[cam_pose_id]
    filenames_fopen = open(filenames_file, 'ab')
    list_filenames_fopen.append(filenames_fopen)

# Camera model instance 
cam_model = CamModel(N=N)

with h5py.File(hfile_dir, 'r') as hf:
  for i in range(start_indx, end_indx):
    progress_bar(i, end_indx- start_indx +1, '====> Iter%d'%i)
    number = str(i)  
    zero = hf[number]
    digit = (zero["img"][:], zero["points"][:], zero.attrs["label"])
    # Point cloud 
    pc = digit[1]  # Numpy, 25700 x 3 
    # Label 
    label = digit[2] # Numpy int64 
    print('===> Label:', label)    


    # Original Image 
    orig_img = digit[0] # Numpy 30 x 30, float64 
    # Save original 2D image 
    orig_two_D_file = os.path.join(origin_two_D_dir, str(i) + '.npy')
    np.save(orig_two_D_file, orig_img)
    
    # Display image  
    if visual:
        disp_image(orig_img) 
  
    # Create voxel 
    voxel_grid = VoxelGrid(pc, x_y_z=[N, N, N], bb_cuboid=False)
    

    # Project points into 2D image 
    rgb = np.ones([N*N*N, 3])
    indices = np.where(voxel_grid.vector.reshape(-1)>0) 
    rgb[indices] = (0.0, 0.0, 0.0) 

    for cam_pose_id in range(len(list_poses)):
        two_D_dir = list_two_D_dirs[cam_pose_id]
        three_D_dir = list_three_D_dirs[cam_pose_id]
        filenames_fopen = list_filenames_fopen[cam_pose_id]
        # Obtain 
        # bbox return has format: [x_min, x_max, y_min, y_max]
        two_D_img, three_D_img, bbox = cam_model.project_3d_2d(rgb, pose=list_poses[cam_pose_id], visual=visual) 
        # Save 
        two_D_file = os.path.join(two_D_dir, str(i) + '.npy')
        three_D_file = os.path.join(three_D_dir, str(i) + '.npy')
        np.save(two_D_file, two_D_img)
        np.save(three_D_file, three_D_img)

        writing_list = [str(i), str(label)] + [str(l) for l in bbox]
        for item in writing_list:
            print('--writing', item)
            filenames_fopen.write(item)
            filenames_fopen.write(' ')
        filenames_fopen.write('\n')

    
# Close all fopen 
for cam_pose_id in range(len(list_poses)):
    filenames_fopen = list_filenames_fopen[cam_pose_id]
    filenames_fopen.close()
    print('===> Closed all filenames_files!')