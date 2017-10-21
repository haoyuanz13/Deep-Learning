# File Classifications

1. model1_torch.py 
------------------
Builds a basic model for object classification, 3 Conv layers and 2 Fully connected layers, includes batch normalization and average pooing. 

2. 'model2_torch.py' 
------------------------
Builds the model based on the first model, but convert all fully connected layers to convolutional ones, also includes 20 more addition layers to check vanishing problem. Then take a reference to the ResNet structure to deal with the deading gradients.

3. 'AdvImg_torch.py'
------------------------
Generate the adversarial images to fool the network. 

4. 'trainAndTest_torch.py'
--------------------------
The main file to train models using cifar10 data, also includes the test scripts. I also created a function to visualize the feature map of different channels.

5. 'utils.py' 
-------------
Contains helper functions.
