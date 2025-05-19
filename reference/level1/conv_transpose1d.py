import torch
import torch.nn as nn

def conv_transpose1d_reference(x, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
    conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 64
    out_channels = 3
    kernel_size = 3
    length = 128
    x = torch.randn(batch_size, in_channels, length)
    return [x, in_channels, out_channels, kernel_size, 1, 0, 0, 1, False]  # Include all parameters
