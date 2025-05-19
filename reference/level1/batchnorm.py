import torch
import torch.nn as nn

def batchnorm_reference(input, num_features):
    bn = nn.BatchNorm2d(num_features=num_features)
    return bn(input)

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x, features]  # Include num_features
