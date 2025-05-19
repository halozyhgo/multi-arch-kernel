import torch
import torch.nn as nn

def maxpool3d_reference(input, kernel_size=3, stride=2, padding=1, dilation=3, return_indices=False, ceil_mode=False):
    return nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(input)

def get_inputs():
    batch_size = 16
    channels = 32
    dim1 = 64
    dim2 = 64
    dim3 = 64
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x, 3, 2, 1, 3, False, False]  # Include all parameters
