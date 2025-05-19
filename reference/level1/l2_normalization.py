import torch

def l2_normalization_reference(x):
    return x / torch.norm(x, p=2, dim=1, keepdim=True)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]