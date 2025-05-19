import torch
import torch.nn as nn

def layernorm_reference(input, normalized_shape):
    ln = nn.LayerNorm(normalized_shape=normalized_shape)
    return ln(input)

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x, (features, dim1, dim2)]  # Include normalized_shape
