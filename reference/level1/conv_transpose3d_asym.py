import torch
import torch.nn as nn

def conv_transpose3d_asym_reference(x, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
    conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 5, 7)  # Asymmetric kernel
    depth = 16
    height = 32
    width = 64  # Asymmetric input
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x, in_channels, out_channels, kernel_size, (1,1,1), (0,0,0), (0,0,0), 1, False]  # Include all parameters
