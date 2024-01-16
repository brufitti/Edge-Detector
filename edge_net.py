"""
Convolutional Neural Network for edge detecting
"""
import torch
import torch.nn as nn

class EdgeDetector(nn.Module):
    def __init__(self, inChannels=1, outChannels=1, kernelSize=3, stride=1, bias=False):
        super(EdgeDetector, self).__init__()
        
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = bias
        
        #layers
        self.convolution = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, image):
        output = self.convolution(image)
        return output
    