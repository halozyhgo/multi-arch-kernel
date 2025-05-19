import torch

def hinge_loss_reference(predictions, targets):
    """
    Computes Hinge Loss for binary classification.

    Args:
        predictions (torch.Tensor): Model predictions (-∞ to ∞)
        targets (torch.Tensor): Ground truth labels (-1 or 1)

    Returns:
        torch.Tensor: Hinge loss value
    """
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))

def get_inputs():
    batch_size = 128
    input_shape = (1,)
    return [
        torch.randn(batch_size, *input_shape),  # predictions
        torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1  # targets (-1 or 1)
    ]
