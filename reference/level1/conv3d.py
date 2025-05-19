import torch
import torch.nn as nn

def conv3d_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    depth = 64
    width = 64
    height = 64
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x, in_channels, out_channels, kernel_size, 1, 0, 1, 1, False]  # Include all parameters
