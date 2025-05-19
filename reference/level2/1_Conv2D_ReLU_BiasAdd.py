import torch

def conv_relu_bias_reference(x, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs convolution, applies ReLU, and adds bias.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W)
        weight (torch.Tensor): Convolution weights of shape (C_out, C_in, kH, kW)
        bias (torch.Tensor): Bias tensor of shape (C_out, 1, 1)
        stride, padding, dilation, groups: Convolution parameters

    Returns:
        torch.Tensor: Output tensor after convolution, ReLU and bias addition
    """
    x = torch.nn.functional.conv2d(x, weight, None, stride, padding, dilation, groups)
    x = torch.relu(x)
    x = x + bias
    return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 16
    height, width = 32, 32
    kernel_size = 3
    
    x = torch.randn(batch_size, in_channels, height, width)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels, 1, 1)
    
    return [x, weight, bias]
