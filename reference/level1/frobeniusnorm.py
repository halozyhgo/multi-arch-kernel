import torch

def frobeniusnorm_reference(input):
    norm = torch.norm(input, p='fro')
    return input / norm

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]
