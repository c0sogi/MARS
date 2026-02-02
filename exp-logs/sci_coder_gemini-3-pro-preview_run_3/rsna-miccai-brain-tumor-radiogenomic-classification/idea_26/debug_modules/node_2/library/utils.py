import torch
import numpy as np
import random
from library.config import set_seed, DEVICE


def seed_everything(seed: int) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation in library.config.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def get_device() -> torch.device:
    """
    Returns the computing device (GPU or CPU) defined in the configuration.

    Returns:
        torch.device: The device object.
    """
    return DEVICE


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
