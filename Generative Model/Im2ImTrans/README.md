Image to Image translation with Conditional Adversarial Networks
========================================

Tensorflow implementation of the [Image-to-Image Translation Using Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf), which learns a mapping(transformation) from the input sketches(e.g. labels, edges or aerial) to the output photos.

## Introduction
In this work, I completed the overall network construction, and used two datasets, CUFS Students Faces and Facades, to train the model. Below sections show more details about the dataset, model architectures and package executions, also include the experimental results to verify the accuracy of my performance.

## Package Clarification
### Data
In this work, I use two types of datasets to train the model, each of them gave me the promising results.
1. [_CUFS_Students_](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html): includes the human face sketches and the real photos, total **88** training and **100** testing instances respectively.

2. [_Facades_](http://cmp.felk.cvut.cz/~tylecr1/facade/): presents a dataset of facade images assembled at the Center for Machine Perception, which includes **606** rectified images of facades from various sources, which have been manually annotated. The facades are from different cities around the world and diverse architectural styles. 

### Models
1. _img2img_x_          
2. _img2img_
### Other Files

## Code Execution

## Experiments
