import torch
import torch.nn as nn

def triplet_margin_reference(anchor, positive, negative, margin=1.0):
    return nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)

def get_inputs():
    batch_size = 128
    input_shape = (4096, )
    return [torch.randn(batch_size, *input_shape), 
            torch.randn(batch_size, *input_shape), 
            torch.randn(batch_size, *input_shape),
            1.0]  # Include margin parameter
