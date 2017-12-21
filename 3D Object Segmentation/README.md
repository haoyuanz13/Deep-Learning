# 3D object classification, segmentation and 3D bbox estimation               

Please feel free to check the detailed report: [3D object recognition and 3D bbox estimation](./res/3d_objectDetection_report.pdf)           


The main contributions
----------------------
1. 3D Data (3D MNIST) Pre-Processing, including the Voxel generation and 2D multiple views generation.
2. Deep network construction which can estimate class label, depth map and 2D bbox in one single stage.
3. 3D bounding box estimation.



The whole model diagram
-----------------------
<p >
<align="center">
  <img src = "./doc/model.png?raw=true" width="650" height="400">
</p>

The deep network architecture
-----------------------------
<p >
<align="center">
  <img src = "./doc/network.jpg?raw=true" width="900" height="450">
</p>

Explanations on deep network               

1. Multiple views share the same Basenet to extract features, which will return a list of view.      

2. Separate into three branches.                  
   - cls: first flatten feature map into vectors, then view max pooling and concatenate into one feature vector finally. Softmax and cross entropy to get cls loss.                 
   - reg: use a 4d vector [x_min, x_max, y_min, y_max] to represent single bbox. Flatten feature maps, then feed into the Fully connected block, all views share the same FC block (the similar idea as the sharing Basenet which can make the FC block handle all views information), finally compute the l2 loss.                
   - depth: use the Deconv Block (the similar strategy as semantic segmentation) which can keep the spatial information, then implement the pixelwise loss computation, only consider the depth information locate in 2D bbox region (ignore backgrounds) for the fast training, also provide better performance.            
   
3. Weighted-sum loss (Multi-task loss) to train the three branches in one single state.           

Training and Test Results
-------------------------

<p >
<align="center">
  <img src = "./res/cls_2000_res.png?raw=true" width="800" height="400">
</p>
  
<p >
<align="center">
  <img src = "./res/reg_2000_res.png?raw=true" width="800" height="400">
</p>

<p >
<align="center">
  <img src = "./res/depth_2000_res.png?raw=true" width="800" height="400">
</p>


Estimated Results
-----------------
<p >
<align="letf">
  <img src = "./doc/3dbboxDigit4.png?raw=true" width="400" height="350">
<align="rightr">
  <img src = "./doc/3dbboxDigit5.png?raw=true" width="400" height="350">
</p>
