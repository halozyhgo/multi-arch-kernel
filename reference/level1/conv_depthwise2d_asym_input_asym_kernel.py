import torch
import torch.nn as nn

def conv_depthwise2d_asym_input_asym_kernel_reference(x, in_channels, kernel_size_h, kernel_size_w, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1, bias=False):
    conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w),
                    stride=(stride_h, stride_w), padding=(padding_h, padding_w),
                    dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)
    return conv(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    kernel_size_h = 3  # Asymmetric kernel dimensions
    kernel_size_w = 5
    height = 128  # Asymmetric input dimensions
    width = 256
    x = torch.randn(batch_size, in_channels, height, width)
    return [x, in_channels, kernel_size_h, kernel_size_w, 1, 1, 0, 0, 1, 1, False]  # Include all parameters
