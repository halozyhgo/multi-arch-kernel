import torch
import torch.nn.functional as F

def kldiv_reference(predictions, targets):
    return F.kl_div(torch.log(predictions), targets, reduction='batchmean')

def get_inputs():
    batch_size = 128
    input_shape = (4096, )
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), 
            torch.randn(batch_size, *input_shape).softmax(dim=-1)]
