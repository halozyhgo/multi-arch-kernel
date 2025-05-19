import torch
def square_matrix_multiplication_reference(A, B):
    return torch.matmul(A, B)

def get_inputs():
    M = 1024
    K = 4096
    N = 2048
    return [torch.randn(M,K), torch.randn(K,N)]