import torch
import torch.nn as nn

def avgpool3d_reference(input, kernel_size=3, stride=2, padding=1):
    return nn.AvgPool3d(kernel_size, stride, padding)(input)

def get_inputs():
    batch_size = 16
    channels = 32
    depth = 64
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x, 3, 2, 1]  # Include all parameters
