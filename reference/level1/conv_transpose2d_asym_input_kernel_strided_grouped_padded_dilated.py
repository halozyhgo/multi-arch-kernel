import torch
import torch.nn as nn

def conv_transpose2d_asym_input_kernel_strided_grouped_padded_dilated_reference(x, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, dilation=dilation,
                            groups=groups, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5)  # Asymmetric kernel
    height = 128
    width = 256  # Asymmetric input dimensions
    stride = (2, 3)  # Strided
    padding = (1, 2)  # Padded
    dilation = (2, 1)  # Dilated
    groups = 4  # Grouped
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False]  # Include all parameters
