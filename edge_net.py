"""
Convolutional Neural Network for edge detecting
"""
import torch
import torch.nn as nn

class EdgeDetector(nn.Module):
    def __init__(self, inShape=(10,10,1), outShape=(10,10,1), kernelSize=3, stride=1, bias=False):
        super(EdgeDetector, self).__init__()
        
        # Shape = Height, Width, Channels
        self.inHeight = inShape[0]
        self.inWidth = inShape[1]
        self.inChannels = inShape[2]
        self.outHeight = outShape[0]
        self.outWidth = outShape[1]
        self.outChannels = outShape[2]
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = bias
        
        #layers
        # self.convolution = torch.nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        # )
        self.cnn = nn.Conv2d(in_channels=self.inChannels, out_channels=self.outChannels, kernel_size=self.kernelSize, stride=self.stride, padding=1, bias=self.bias)
        self.ReLu = torch.nn.ReLU()
    def forward(self, image):
        # output = self.convolution(image)
        output = self.cnn(image)
        output = self.ReLu(output)
        return output
    