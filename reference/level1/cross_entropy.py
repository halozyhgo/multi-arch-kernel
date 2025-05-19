import torch
import torch.nn.functional as F

def cross_entropy_reference(predictions, targets):
    return F.cross_entropy(predictions, targets)

def get_inputs():
    batch_size = 4096
    num_classes = 10
    input_shape = (num_classes, )  # Output for each class
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]
