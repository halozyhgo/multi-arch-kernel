import torch
import torch.nn.functional as F

def cosine_similarity_loss_reference(predictions, targets):
    """
    Computes Cosine Similarity Loss between predictions and targets.

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values

    Returns:
        torch.Tensor: Cosine similarity loss value (1 - cosine similarity)
    """
    cosine_sim = F.cosine_similarity(predictions, targets, dim=1)
    return torch.mean(1 - cosine_sim)

def get_inputs():
    batch_size = 128
    input_shape = (4096,)
    predictions = torch.randn(batch_size, *input_shape)
    targets = torch.randn(batch_size, *input_shape)
    return [predictions, targets]
