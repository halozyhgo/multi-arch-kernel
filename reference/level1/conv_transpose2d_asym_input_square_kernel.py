import torch
import torch.nn as nn

def conv_transpose2d_asym_input_square_kernel_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3  # Square kernel
    height_in = 128
    width_in = 256  # Asymmetric input dimensions
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x, in_channels, out_channels, kernel_size, 1, 0, 0, 1, False]  # Include all parameters
