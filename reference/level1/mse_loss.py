import torch

def mse_loss_reference(predictions, targets):
    """
    Computes Mean Squared Error loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values

    Returns:
        torch.Tensor: MSE loss value
    """
    return torch.mean((predictions - targets) ** 2)

def get_inputs():
    batch_size = 128
    input_shape = (4096,)
    predictions = torch.randn(batch_size, *input_shape)
    targets = torch.randn(batch_size, *input_shape)
    return [predictions, targets]
