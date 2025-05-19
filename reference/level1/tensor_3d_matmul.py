import torch

def tensor_3d_matmul_reference(A, B):
    """
    Performs 3D tensor-matrix multiplication.

    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K)
        B (torch.Tensor): Input matrix of shape (K, L)

    Returns:
        torch.Tensor: Output tensor of shape (N, M, L)
    """
    return torch.matmul(A, B)

def get_inputs():
    N = 16
    M = 1024
    K = 2048
    L = 768
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return [A, B]
