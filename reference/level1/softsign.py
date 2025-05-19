import torch
def softsign_reference(input):
    return input / (1 + torch.abs(input))

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
