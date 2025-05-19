import torch

def matrix_scalar_mul_reference(A, s):
    """
    Performs matrix-scalar multiplication (C = A * s).

    Args:
        A (torch.Tensor): Input matrix of shape (M, N)
        s (float): Scalar value

    Returns:
        torch.Tensor: Resulting matrix of shape (M, N)
    """
    return A * s

def get_inputs():
    M = 16384
    N = 4096
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]
