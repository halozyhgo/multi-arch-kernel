import torch

def rms_norm_reference(input, num_features, eps=1e-5):
    # Calculate the RMS along the feature dimension
    rms = torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + eps)
    # Normalize the input by dividing by the RMS
    return input / rms

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x, features]  # Include num_features
