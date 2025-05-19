import torch
import torch.nn as nn

def conv_transpose2d_asym_input_kernel_padded_reference(x, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=False):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5)  # Asymmetric kernel
    height = 128
    width = 256  # Asymmetric input dimensions
    padding = (1, 2)  # Padded
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, kernel_size, (1, 1), padding, False]  # Include all parameters
