import torch
import torch.nn as nn

def conv_depthwise2d_asym_input_square_kernel_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                    stride=stride, padding=padding, groups=in_channels, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 3
    kernel_size = 3  # Square kernel
    height = 128  # Asymmetric input dimensions
    width = 256
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, kernel_size, 1, 0, False]  # Include all parameters
