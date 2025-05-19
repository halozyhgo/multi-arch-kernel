import torch

def masked_cumsum_reference(x, mask, dim):
    """
    Computes masked cumulative sum along the specified dimension (only sums elements where mask is True).

    Args:
        x (torch.Tensor): Input tensor
        mask (torch.Tensor): Boolean mask tensor
        dim (int): Dimension along which to compute masked cumulative sum

    Returns:
        torch.Tensor: Tensor with masked cumulative sum along specified dimension
    """
    return torch.cumsum(x * mask, dim=dim)

def get_inputs():
    batch_size = 128
    input_shape = (4000,)
    dim = 1  # Dimension for masked cumsum
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask, dim]
