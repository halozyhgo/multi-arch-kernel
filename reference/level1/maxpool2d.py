import torch
import torch.nn as nn

def maxpool2d_reference(input, kernel_size=2, stride=2, padding=1, dilation=3):
    return nn.MaxPool2d(kernel_size, stride, padding, dilation)(input)

def get_inputs():
    batch_size = 16
    channels = 32
    height = 128
    width = 128
    x = torch.randn(batch_size, channels, height, width)
    return [x, 2, 2, 1, 3]  # Include all parameters
