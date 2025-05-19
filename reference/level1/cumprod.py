import torch

def cumprod_reference(x, dim):
    """
    Computes cumulative product along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute cumulative product

    Returns:
        torch.Tensor: Tensor with cumulative product along specified dimension
    """
    return torch.cumprod(x, dim=dim)

def get_inputs():
    batch_size = 128
    input_shape = (4000,)  # Example shape
    dim = 1  # Dimension for cumprod
    x = torch.randn(batch_size, *input_shape)
    return [x, dim]
