import torch
import torch.nn as nn

def conv_transpose3d_asym_input_square_kernel_strided_grouped_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
    conv = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size),
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3  # Square kernel
    depth = 16
    height = 32
    width = 32  # Asymmetric input dimensions
    stride = 2  # Strided
    padding = 3  # Padded
    groups = 4  # Grouped
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x, in_channels, out_channels, kernel_size, stride, padding, 0, groups, False]  # Include all parameters
