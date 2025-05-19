import torch
def leaky_relu_reference(input, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(input, negative_slope=negative_slope)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
