import torch
import torch.nn as nn

def conv_transpose2d_asym_kernel_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5)  # Asymmetric kernel
    height = 128
    width = 128  # Square input dimensions
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, kernel_size, 1, 0, 0, 1, False]  # Include all parameters
