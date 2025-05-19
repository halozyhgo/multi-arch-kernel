import torch
import torch.nn as nn

def maxpool1d_reference(input, kernel_size=4, stride=2, padding=2, dilation=3, return_indices=False):
    return nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices)(input)

def get_inputs():
    batch_size = 16
    features = 64
    sequence_length = 128
    x = torch.randn(batch_size, features, sequence_length)
    return [x, 4, 2, 2, 3, False]  # Include all parameters
