import torch

def cumsum_reverse_reference(x, dim):
    """
    Computes reverse cumulative sum along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute reverse cumulative sum

    Returns:
        torch.Tensor: Tensor with reverse cumulative sum along specified dimension
    """
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)

def get_inputs():
    batch_size = 128
    input_shape = (4000,)  # Example shape
    dim = 1  # Dimension for reverse cumsum
    x = torch.randn(batch_size, *input_shape)
    return [x, dim]
