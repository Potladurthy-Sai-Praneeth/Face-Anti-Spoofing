import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from depth_map_model import FineTuneDepthAnything


class CDC(nn.Module):
    '''
    This class performs central difference convolution (CDC) operation. 
    First the normal convolution is performed and then the convolution is performed with the squeezed version of kernel and the difference is the result.
    Args :
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The size of the kernel.
        stride : int
            The stride of the convolution.
        padding : int
            The padding of the convolution.
        dilation : int
            The dilation of the convolution.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias or not.
        theta : float
            The value of theta.
    Returns :
        out_normal -  torch.tensor
            The resultant image/channels of the CDC operation.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC, self).__init__()
        self.bias= bias
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.theta = theta
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding if kernel_size==3 else 0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        # if conv.weight is (out_channels, in_channels, kernel_size, kernel_size),
        # then the  self.conv.weight.sum(2) will return (out_channels, in_channels,kernel_size)
        # and self.conv.weight.sum(2).sum(2) will return (out_channels,n_channels)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        # Here we are adding extra dimensions such that the kernel_diff is of shape (out_channels, in_channels, 1, 1) so that convolution can be performed.
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, stride=self.stride, padding=0, groups=self.groups)
        return out_normal - self.theta * out_diff
    

class conv_block_nested(nn.Module):
    def __init__(self, in_ch,  out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = CDC(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = CDC(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class ClassifierUCDCN(nn.Module):
    def __init__(self, dropout=0.5):
        super(ClassifierUCDCN, self).__init__()        
        self.layers =8
        self.dropout_prob = dropout
        self.img_size = (252, 252)
        self.hidden_size = 64
        self.conv1 = conv_block_nested(1,self.layers)
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = conv_block_nested(self.layers,1)
        # Maxpool
        self.linear_1 = nn.Linear((self.img_size[0]//4 * self.img_size[1]//4), self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear_2 = nn.Linear(self.hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inp):
        conv1 = self.conv1(inp)
        maxpool = self.maxpool(conv1)
        conv2 = self.conv2(maxpool)
        maxpool2 = self.maxpool(conv2)
        linear_1 = self.linear_1(maxpool2.view(-1, self.img_size[0]//4 * self.img_size[1]//4))
        relu = self.relu(linear_1)
        dropout = self.dropout(relu)
        linear_2 = self.linear_2(dropout)
        return self.sigmoid(linear_2)