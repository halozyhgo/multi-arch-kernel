import torch
import torch.nn.functional as F

def huber_loss_reference(predictions, targets):
    """
    Computes Huber loss (smooth L1 loss) between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values

    Returns:
        torch.Tensor: Huber loss value
    """
    return F.smooth_l1_loss(predictions, targets)

def get_inputs():
    batch_size = 128
    input_shape = (4096,)
    predictions = torch.randn(batch_size, *input_shape)
    targets = torch.randn(batch_size, *input_shape)
    return [predictions, targets]
