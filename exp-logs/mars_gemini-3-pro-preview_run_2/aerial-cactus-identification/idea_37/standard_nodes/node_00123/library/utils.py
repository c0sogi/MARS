import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in cuDNN backends.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode is slower but required for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuses a Conv2d layer and a BatchNorm2d layer into a single Conv2d layer
    for inference efficiency (Structural Re-parameterization).

    Mathematically transforms:
        y = (conv(x) - mean) / sqrt(var + eps) * gamma + beta
    Into:
        y = fused_conv(x) + fused_bias

    Args:
        conv (nn.Conv2d): The convolutional layer.
        bn (nn.BatchNorm2d): The batch normalization layer.

    Returns:
        nn.Conv2d: The fused convolutional layer with bias.
    """
    # 1. Create a new Conv2d layer with the same configuration
    # The fused layer will always have a bias term to absorb BN's beta and running mean
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    # 2. Ensure the new layer is on the same device as the original
    device = conv.weight.device
    fused_conv = fused_conv.to(device)

    # 3. Extract weights and BN statistics
    # Clone to prevent side effects on the original model
    w = conv.weight.clone()
    mean = bn.running_mean.clone()
    var_sqrt = torch.sqrt(bn.running_var.clone() + bn.eps)

    beta = bn.bias.clone()
    gamma = bn.weight.clone()

    if conv.bias is not None:
        b = conv.bias.clone()
    else:
        # If original conv has no bias, treat it as zeros
        b = torch.zeros_like(mean)

    # 4. Calculate fused weights and bias
    # Reshape scale factor to match weight dimensions for broadcasting
    # w shape: (out_channels, in_channels // groups, k, k)
    # scale shape needed: (out_channels, 1, 1, 1)
    scale = gamma / var_sqrt
    reshape_scale = scale.view(-1, 1, 1, 1)

    fused_weight = w * reshape_scale
    fused_bias = (b - mean) * scale + beta

    # 5. Assign parameters to the new layer
    fused_conv.weight = nn.Parameter(fused_weight)
    fused_conv.bias = nn.Parameter(fused_bias)

    return fused_conv
