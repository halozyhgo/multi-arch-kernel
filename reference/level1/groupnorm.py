import torch
import torch.nn as nn

def groupnorm_reference(input, num_features, num_groups):
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    return gn(input)

def get_inputs():
    batch_size = 16
    features = 64
    num_groups = 8
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x, features, num_groups]  # Include num_features and num_groups
