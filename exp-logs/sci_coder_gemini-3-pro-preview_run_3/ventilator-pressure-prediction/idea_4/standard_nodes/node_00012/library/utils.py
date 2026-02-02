import torch
import os
import sys

# Import the pre-defined configuration and seeding function
# to avoid re-implementation and ensure consistency.
from library.config import set_seed


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized implementation in library.config.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    set_seed(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Automatically selects the computing device (GPU if available, else CPU).

    Args:
        force_cpu (bool): If True, forces the use of CPU even if GPU is available.

    Returns:
        torch.device: The selected device.
    """
    if not force_cpu and torch.cuda.is_available():
        # In a multi-GPU environment, this defaults to the current device (usually index 0)
        return torch.device("cuda")
    else:
        return torch.device("cpu")
