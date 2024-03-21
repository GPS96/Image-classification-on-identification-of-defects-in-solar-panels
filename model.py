import torch
from torch.nn import Module, Conv2d, BatchNorm2d, MaxPool2d, ReLU, Flatten, AvgPool2d, Sigmoid, Linear, Sequential, functional, Dropout
from torch.utils.data import DataLoader
from data import ChallengeDataset
import pandas as pd
import numpy as np

class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.batch_dim = None

    def forward(self, input_tensor):
        self.batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(self.batch_dim, -1)

class ResBlock(Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(ResBlock,self).__init__()
        self.conv2d_1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride_shape, padding=1)
        self.BatchNorm_1= BatchNorm2d(out_channels)
        self.Relu_1= ReLU()
        self.conv2d_2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.BatchNorm_2= BatchNorm2d(out_channels)

        self.residual_connection= True
        self.conv_one = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride_shape)

        #Setting up the condition for residual connection to skip some layers
        if in_channels == out_channels and stride_shape == 1:
            self.residual_connection= False
        else:
            self.residual_connection= True

        self.BatchNorm_3= BatchNorm2d(out_channels)
        self.Relu_2= ReLU()
        self.Sequence= Sequential(self.conv2d_1, self.BatchNorm_1, self.Relu_1, self.conv2d_2, self.BatchNorm_2)
        self.Residual= None

    def forward(self, input_tensor):
        self.Residual= input_tensor
        output_tensor= self.Sequence(input_tensor)
        if self.residual_connection == True:
            self.Residual= self.conv_one(self.Residual)

        #Now we do batch normalization for the skip path
        self.Residual= self.BatchNorm_3(self.Residual)
        output_tensor += self.Residual
        output_tensor= self.Relu_2(output_tensor)

        return output_tensor

class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()

        #Setting up the sequence of layers in ResNet
        self.conv2d = Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.BatchNorm = BatchNorm2d(num_features=64)
        self.Relu = ReLU()
        self.Maxpool = MaxPool2d(kernel_size=3, stride=2)
        self.Sequence_ResNet= Sequential(self.conv2d, self.BatchNorm, self.Relu, self.Maxpool)

        #Joining the ResBlocks
        self.ResBlock_1 = ResBlock(in_channels=64, out_channels=64)
        self.ResBlock_2 = ResBlock(in_channels=64, out_channels=128, stride_shape=2)
        self.ResBlock_3 = ResBlock(in_channels=128, out_channels=256, stride_shape=2)
        Dropout(p=0.5)
        self.ResBlock_4 = ResBlock(in_channels=256, out_channels=512, stride_shape=2)
        self.Sequence_ResBlock= Sequential(self.ResBlock_1, self.ResBlock_2, self.ResBlock_3, Dropout(), self.ResBlock_4)

        #Computation of results from the combined ResBlocks
        self.GlobalAvgPool = AvgPool2d(kernel_size=10)
        self.Flatten = Flatten()
        Dropout(p=0.5)
        self.Fully_Connected = Linear(in_features=512, out_features=2)
        self.Sigmoid = Sigmoid()
        self.Sequence_Computation= Sequential(self.GlobalAvgPool, self.Flatten, Dropout(), self.Fully_Connected, self.Sigmoid)

    def forward(self, input_tensor):

        output_tensor= self.Sequence_ResNet(input_tensor)
        output_tensor= self.Sequence_ResBlock(output_tensor)
        output_tensor= self.Sequence_Computation(output_tensor)

        return output_tensor






