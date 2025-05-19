import torch

def cumsum_reference(x, dim):
    """
    Computes cumulative sum (prefix sum) along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute cumulative sum

    Returns:
        torch.Tensor: Tensor with cumulative sum along specified dimension
    """
    return torch.cumsum(x, dim=dim)

def get_inputs():
    batch_size = 128
    input_shape = (4000,)  # Example shape
    dim = 1  # Dimension for cumsum
    x = torch.randn(batch_size, *input_shape)
    return [x, dim]
