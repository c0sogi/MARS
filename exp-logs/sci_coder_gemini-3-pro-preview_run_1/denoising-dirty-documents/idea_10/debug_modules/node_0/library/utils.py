import os
import sys
import numpy as np
import torch

# Import seed_everything from the provided library configuration file
from library.config import seed_everything


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the library.config.seed_everything function.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def invert_intensity(x):
    """
    Inverts the pixel intensity of the input: new_val = 1.0 - old_val.

    This is used to align the signal such that text (originally 0/black) becomes
    1 (signal) and background (originally 1/white) becomes 0 (sparse).
    This ensures that zero-padding in CNNs acts as background extension
    rather than introducing artifacts.

    Args:
        x (numpy.ndarray or torch.Tensor): Input data with values in range [0, 1].

    Returns:
        numpy.ndarray or torch.Tensor: The inverted data.
    """
    return 1.0 - x


def revert_intensity(x):
    """
    Reverts the inverted intensity back to the original space: original_val = 1.0 - inverted_val.

    This is used to transform the model's predictions back to the original
    grayscale format (0=black text, 1=white background) for submission.

    Args:
        x (numpy.ndarray or torch.Tensor): Input data with values in range [0, 1].

    Returns:
        numpy.ndarray or torch.Tensor: The reverted data.
    """
    return 1.0 - x
