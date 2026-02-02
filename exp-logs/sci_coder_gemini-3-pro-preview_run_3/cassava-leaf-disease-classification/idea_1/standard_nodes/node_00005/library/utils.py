import torch
import numpy as np
import random
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to be set. Defaults to Config.SEED.
    """
    # Update the seed in Config to ensure consistency if accessed elsewhere
    Config.SEED = seed

    # Use the pre-implemented method in Config to set seeds across libraries
    Config.set_seed()


def get_device() -> torch.device:
    """
    Automatically selects the available computing device (GPU or CPU).

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    # Use the pre-implemented method in Config to determine the device
    return Config.get_device()
