import torch
import torch.nn as nn

def conv_transpose3d_asym_kernel_reference(x, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
    conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, output_padding=output_padding,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5, 5)  # Asymmetric kernel (depth differs from width/height)
    depth = 64
    width = 64
    height = 64  # Square input dimensions
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x, in_channels, out_channels, kernel_size, (1,1,1), (0,0,0), (0,0,0), 1, False]  # Include all parameters
