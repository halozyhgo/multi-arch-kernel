import torch
import torch.nn as nn

def conv2d_asym_reference(x, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5)  # Asymmetric kernel
    height = 256
    width = 128  # Asymmetric input
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, kernel_size, (1,1), (0,0), (1,1), 1, False]  # Include all parameters
