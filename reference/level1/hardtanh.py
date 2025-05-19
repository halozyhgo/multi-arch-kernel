import torch
def hardtanh_reference(input, min_val=-1., max_val=1.):
    return torch.nn.functional.hardtanh(input, min_val=min_val, max_val=max_val)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
