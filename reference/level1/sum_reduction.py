import torch

def sum_reduction_reference(x, dim=1, keepdim=True):
    return torch.sum(x, dim=dim, keepdim=keepdim)

def get_inputs():
    batch_size = 16
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, dim1, dim2)
    return [x, 1, True]  # Include all parameters
