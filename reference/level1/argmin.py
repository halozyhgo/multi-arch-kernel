import torch

def argmin_reference(x, dim=1):
    return torch.argmin(x, dim=dim)

def get_inputs():
    batch_size = 16
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, dim1, dim2)
    return [x, 1]  # Include all parameters
