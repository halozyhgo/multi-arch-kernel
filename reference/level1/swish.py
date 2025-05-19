import torch
def swish_reference(input):
    return input * torch.sigmoid(input)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
