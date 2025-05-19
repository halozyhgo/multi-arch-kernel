import torch
import torch.nn as nn

def conv_transpose3d_square_input_kernel_padded_dilated_strided_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
    conv = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size),
                            stride=stride, padding=padding, dilation=dilation, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3  # Square kernel
    depth = 16
    height = 32
    width = 32  # Square input dimensions
    stride = 2  # Strided
    padding = 1  # Padded
    dilation = 2  # Dilated
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x, in_channels, out_channels, kernel_size, stride, padding, dilation, False]  # Include all parameters
