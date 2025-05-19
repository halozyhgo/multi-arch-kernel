import torch
import torch.nn as nn

def avgpool2d_reference(input, kernel_size=3, stride=None, padding=0):
    return nn.AvgPool2d(kernel_size, stride, padding)(input)

def get_inputs():
    batch_size = 16
    channels = 64
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)
    return [x, 3, None, 0]  # Include all parameters
