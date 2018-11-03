# Facial Keypoint Detection
*Computer Vision Nanodegree at Udacity*

Use Pytorch to create a CNN that is able to predict facial keypoints on an image.


## Requirements
* [Environment Setup](https://github.com/udacity/CVND_Exercises)


### The CNN Architecture

The CNN architecture that I used was influenced by [NaimishNet](https://arxiv.org/pdf/1710.00977.pdf) and [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). Mostly built by trial and error because the main purpose was to get some insights about CNN, its components and how  a framework like PyTorch can do the heavy lifting to build and train a model.

Model: `/saved_models/model_net.pt`

* Convolution | in: 1 | out: 32 | kernel: 10 x 10 | stride: 2
* Relu
* MaxPool     | kernel: 2 

* Convolution | in: 32 | out: 64 | kernel: 5 x 5 | stride: 2
* Relu
* MaxPool     | kernel: 2 

* Convolution | in: 64 | out: 128 | kernel: 5 x 5
* Relu
* MaxPool     | kernel: 2 
 
* Convolution | in: 128 | out: 256 | kernel: 3 x 3
* Relu
* MaxPool     | kernel: 2 
 
*  Dense       | in: 256 | out: 256
*  Relu
* Dense       | in: 256 | out: 136


## Image Pre-Processing
I experimented a few variations of the pre-processing step. The majority of the experiments consisted in rescaling the image, performing a random crop and including RGB channels.

* Rescaling the image to 224x224 worked best because:
	* It is the recommend size for this project
	* It is the same size proposed in the AlexNet paper
	* Ground truth values were consistently close to the left eye, the same as observed in the NaimishNet paper
	* All target keypoints were included
* Random cropping was not used as it brought no significant benefits
* Image normalization used default implementation of Normalize

*Note*: Image augmentation could be use to increase the size and add variation to the image dataset


## Loss & Optimization
A regression function - MSE - was used to minimize the distance between the target and predicted points. This loss function provided good results. L1 and L1 Smooth losses were used but no advantage was observed.

The NaimishNet paper uses Adam optimizer with default parameters, I incorporated this strategy after experimenting with different optmizers, such as SGD, and different learning rates.


## Number of Epochs
I took into consideration the training speed and error margin when training the model. Number of epochs was experimented with 1, 5, 30, 50, 100 and 150 iterations. In the end 150 epochs and batch size of 30 provided good results.
