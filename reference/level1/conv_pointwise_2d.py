import torch
import torch.nn as nn

def conv_pointwise_2d_reference(x, in_channels, out_channels, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    height = 256
    width = 256
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, False]  # Include all parameters
