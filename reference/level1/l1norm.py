import torch

def l1norm_reference(input):
    return input / torch.sum(torch.abs(input), dim=1, keepdim=True)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
