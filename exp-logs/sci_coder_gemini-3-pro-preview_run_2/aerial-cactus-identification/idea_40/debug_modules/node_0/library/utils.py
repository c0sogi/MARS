import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import CHECKPOINT_DIR


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filename: str):
    """
    Saves the training checkpoint to the defined checkpoint directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        filename (str): The name of the file to save (e.g., 'model_seed_0.pth').
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    # We do not print here to keep logs clean as per instructions


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fuses a Conv2d layer and a BatchNorm2d layer into a single set of weights and biases.

    Mathematically:
    Let W be conv weights, b be conv bias.
    Let m, v be BN running mean and var.
    Let gamma, beta be BN learnable params.

    The BN operation is: y = gamma * (x - m) / sqrt(v + eps) + beta
    Expanding: y = (gamma / sqrt(v + eps)) * x + (beta - gamma * m / sqrt(v + eps))

    If x = W * input + b, then:
    y = (gamma / sqrt(v + eps)) * (W * input + b) + (beta - gamma * m / sqrt(v + eps))

    New Weight W' = W * (gamma / sqrt(v + eps))
    New Bias b' = b * (gamma / sqrt(v + eps)) + (beta - gamma * m / sqrt(v + eps))
                = (b - m) * (gamma / sqrt(v + eps)) + beta

    Args:
        conv (nn.Conv2d): The convolution layer.
        bn (nn.BatchNorm2d): The batch normalization layer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The fused weights and biases.
    """
    with torch.no_grad():
        w = conv.weight
        if conv.bias is not None:
            b = conv.bias
        else:
            b = torch.zeros_like(bn.running_mean)

        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias

        # Calculate scaling factor
        scale = gamma / var_sqrt

        # Reshape scale for broadcasting against weight tensor
        # Conv2d weights are [out_channels, in_channels, k, k]
        scale_shape = [1] * len(w.shape)
        scale_shape[0] = -1
        scale_reshaped = scale.view(scale_shape)

        w_fused = w * scale_reshaped
        b_fused = (b - mean) * scale + beta

    return w_fused, b_fused


def pad_1x1_to_3x3_tensor(kernel_1x1: torch.Tensor):
    """
    Pads a 1x1 convolution kernel to a 3x3 kernel.

    Args:
        kernel_1x1 (torch.Tensor): Weights of shape [out_c, in_c, 1, 1]

    Returns:
        torch.Tensor: Weights of shape [out_c, in_c, 3, 3] with the 1x1 content in the center.
    """
    if kernel_1x1 is None:
        return 0

    return torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])
