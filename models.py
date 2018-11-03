import torch
import torch.nn as nn

    
""" 
A CNN architecture influenced by NaimishNet and AlexNet. Mostly built by trial and
error because the main purpose was to get some insights about CNN, its componenets and how 
a framework like PyTorch can do the heavy lifting to build and train a model.

    - Convolution | in: 1 | out: 32 | kernel: 10 x 10 | stride: 2
    - Relu
    - MaxPool     | kernel: 2 

    - Convolution | in: 32 | out: 64 | kernel: 5 x 5 | stride: 2
    - Relu
    - MaxPool     | kernel: 2 

    - Convolution | in: 64 | out: 128 | kernel: 5 x 5
    - Relu
    - MaxPool     | kernel: 2 

    - Convolution | in: 128 | out: 256 | kernel: 3 x 3
    - Relu
    - MaxPool     | kernel: 2 

    - Dense       | in: 256 | out: 256
    - Relu

    - Dense       | in: 256 | out: 136 
    
    References:
        - https://arxiv.org/pdf/1710.00977.pdf
        - https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 10, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 136),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)            
        return x

