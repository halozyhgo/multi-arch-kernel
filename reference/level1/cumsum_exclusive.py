import torch

def cumsum_exclusive_reference(x, dim):
    """
    Computes exclusive cumulative sum along the specified dimension (does not include current element).

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute exclusive cumulative sum

    Returns:
        torch.Tensor: Tensor with exclusive cumulative sum along specified dimension
    """
    exclusive_cumsum = torch.cat((torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x), dim=dim)[:-1]
    return torch.cumsum(exclusive_cumsum, dim=dim)

def get_inputs():
    batch_size = 128
    input_shape = (4000,)
    dim = 1  # Dimension for exclusive cumsum
    x = torch.randn(batch_size, *input_shape)
    return [x, dim]
