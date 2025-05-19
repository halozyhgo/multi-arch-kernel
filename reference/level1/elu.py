import torch
def elu_reference(input, alpha=1.0):
    return torch.nn.functional.elu(input, alpha=alpha)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x, 1.0]  # Include alpha value
