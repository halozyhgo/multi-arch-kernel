import torch
import torch.nn as nn

def avgpool1d_reference(input, kernel_size=4, stride=2, padding=1):
    return nn.AvgPool1d(kernel_size, stride, padding)(input)

def get_inputs():
    batch_size = 16
    in_channels = 32
    input_length = 128
    x = torch.randn(batch_size, in_channels, input_length)
    return [x, 4, 2, 1]  # Include all parameters
