import torch
import torch.nn as nn

def conv1d_dilated_strided_reference(x, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
    conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                    stride=stride, dilation=dilation, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    length = 256
    stride = 3
    dilation = 4
    x = torch.randn(batch_size, in_channels, length)
    return [x, in_channels, out_channels, kernel_size, stride, dilation, False]  # Include all parameters
