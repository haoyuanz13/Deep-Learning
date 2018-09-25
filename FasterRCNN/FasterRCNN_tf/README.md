The intact simplified faster rcnn model for the cifar10 object detection and location task, including RPN and Spatial Transformer function (tf and pure numpy version).

Pkg Clarification
-------------------
1. The **_layers.py_** contains multiple functional layers such as ConvNet, Fully Connected Layers and ResNet Block.    
2. The **_spatial_transformer.py_** includes the spatial transformer function.     
3. The **_fasterRCNN.py_** contains the main structure of Faster RCNN combined with RPN, as well as the executable function for training and test.      
4. The **_helper.py_** file includes all helper functions such as the data loader and the result figure generation.     
