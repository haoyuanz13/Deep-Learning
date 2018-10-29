The intact simplified faster rcnn model for the cifar10 object detection and location task, including RPN and Spatial Transformer function (tf and pure numpy version).

Pkg Clarification
-------------------
1. The `layers.py` contains multiple functional layers such as ConvNet, Fully Connected Layers and ResNet Block.    
2. The `spatial_transformer.py` includes the spatial transformer function.     
3. The `fasterRCNN.py` contains the main structure of Faster RCNN combined with RPN, as well as some executable functions for the train and test tasks.      
4. The `helper.py` file includes all helper functions such as the data loader and the result figure generation.     
