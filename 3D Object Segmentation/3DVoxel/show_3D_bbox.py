import numpy as np
from IPython.display import IFrame
from IPython.display import display 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, pdb
import h5py
from utils import TEMPLATE_POINTS, TEMPLATE_VG, plot_voxelgrid, plot_points, array_to_color, disp_image
from voxelgrid import VoxelGrid
from camera_model import CamModel  
from utils import progress_bar, draw_plane




########################################################################
# Changable parameters 
# Choose start index and end index of data points that we want to compute. 
# This help do data processing in parallel 

start_indx = 0  
end_indx = 2500

# Choose mode 'train' or 'test'
mode = 'train'
resume = True 
visual = True  

# Choose directory where raw data are stored. This contains .h5 files  
raw_dir = "/home/tynguyen/cis680/data/3d-mnist/source"

# Choose target directory 
data_dir = "/home/tynguyen/cis680/data/3d-mnist/target"

# Choose how fine the voxel discretization is 
N = 60 


########################################################################
# Fixed parameters 
# List of camera poses 
list_poses = [[0,  0, 0], [0, 0, 30], [0, 0, 60], [0, 0, 90], 
              [90, 0, 0], 
              [0, 90, 0]] # [Rx, Ry, Rz]

list_colors = ['g','r','m', 'b', 'y', 'c'] 
# Source directories 
hfile_dir = os.path.join(raw_dir, mode + '_point_clouds.h5')

# Target directories 
origin_two_D_dir = os.path.join(data_dir, '2D/' + 'orig/') # Store the original 2D image

# Camera model instance 
cam_model = CamModel(N=N)

with h5py.File(hfile_dir, 'r') as hf:
    for i in range(start_indx, end_indx):
        # if i <= 3:
        #     continue 
        progress_bar(i, end_indx- start_indx +1, '====> Iter %d'%i)
        number = str(i)  
        zero = hf[number]
        digit = (zero["img"][:], zero["points"][:], zero.attrs["label"])
        # Point cloud 
        pc = digit[1]  # Numpy, 25700 x 3 
        # Label 
        label = digit[2] # Numpy int64 
        # Original Image 
        orig_img = digit[0] # Numpy 30 x 30, float64 
        # Save original 2D image 
        orig_two_D_file = os.path.join(origin_two_D_dir, str(i) + '.npy')
       
    
        # Display image  
        if visual:
            disp_image(orig_img) 

        # Create voxel 
        voxel_grid = VoxelGrid(pc, x_y_z=[N, N, N], bb_cuboid=False)
        vector = voxel_grid.vector 
        xx, yy, zz = np.where(vector>0) 

        #print('===> Label:', label) 
        list_3D_planes = []    
        min_xyzs = np.zeros([len(list_poses), 3])
        max_xyzs = np.zeros([len(list_poses), 3])
        for cam_pose_id in range(len(list_poses)):
            # Choose one camera pose ID 
            print('\n===> Camera %d'%cam_pose_id)
            two_D_dir = os.path.join(data_dir, '2D/' + 'pose_' + str(cam_pose_id)) # + '/')
            three_D_dir = os.path.join(data_dir, '3D/' + 'pose_' + str(cam_pose_id)) #) + '/')

            two_D_file = os.path.join(two_D_dir, str(i) +'.npy') 
            two_D_img = np.load(two_D_file)
            three_D_file = os.path.join(three_D_dir, str(i) +'.npy') 
            three_D_img = np.load(three_D_file)

            # Back projection 
            three_D_plane, min_z = cam_model.backproject_2d_3d(three_D_img, pose=list_poses[cam_pose_id], visual=True) # 3xN 
            list_3D_planes.append(three_D_plane)

            # Find min, max x, y, z 
            min_xyz = np.min(three_D_plane, axis=1)
            min_xyz[2] = min_z # Avoid 0 
            max_xyz = np.max(three_D_plane, axis=1)
            min_xyzs[cam_pose_id] = min_xyz
            max_xyzs[cam_pose_id] = max_xyz

            # if visual:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     scatter =  ax.scatter(xx, yy, zz, cmap='coolwarm',linewidth=0, antialiased=False, c='r')
            #     # ax = fig.add_subplot(122, projection='3d')
            #     ax.scatter(three_D_plane[0], three_D_plane[1], three_D_plane[2])
            #     plt.xlabel('x')
            #     plt.ylabel('y')  
            #     plt.show()

        min_xyz = np.min(min_xyzs, 0) # (3,)
        max_xyz = np.max(max_xyzs, 0)
        min_xyz[2] = 0



    

        if visual:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            scatter =  ax.scatter(xx, yy, zz, c='b',linewidth=0, antialiased=False)
             
            # for cam_pose_id in range(len(list_poses)):
            #     three_D_plane = list_3D_planes[cam_pose_id]
            #     ax.scatter(three_D_plane[0], three_D_plane[1], three_D_plane[2], c=list_colors[cam_pose_id], alpha=.55, s=1)
            # plt.show()
              
            # # Draw 3D bbox 
            # Z = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
            #   [max_xyz[0], min_xyz[1], min_xyz[2]],
            #   [max_xyz[0], max_xyz[1], min_xyz[2]],
            #   [min_xyz[0], max_xyz[1], min_xyz[2]],
            #   [min_xyz[0], min_xyz[1], max_xyz[2]],
            #   [max_xyz[0], min_xyz[1], max_xyz[2]],
            #   [max_xyz[0], max_xyz[1], max_xyz[2]],
            #   [min_xyz[0], max_xyz[1], max_xyz[2]]])

            # # Display the bounding boxes
         
            # # plot vertices
            # ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], c='r')

            # # list of sides' polygons of figure
            # verts = [[Z[0],Z[1],Z[2],Z[3]],
            #  [Z[4],Z[5],Z[6],Z[7]], 
            #  [Z[0],Z[1],Z[5],Z[4]], 
            #  [Z[2],Z[3],Z[7],Z[6]], 
            #  [Z[1],Z[2],Z[6],Z[5]],
            #  [Z[4],Z[7],Z[3],Z[0]], 
            #  [Z[2],Z[3],Z[7],Z[6]]]

            # # plot sides
            # ax.add_collection3d(Poly3DCollection(verts, 
            #  facecolors='y', linewidths=1, edgecolors='r', alpha=.15))

            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')




            plt.show()
        
        if i == 4:
            exit()



 

  

    



