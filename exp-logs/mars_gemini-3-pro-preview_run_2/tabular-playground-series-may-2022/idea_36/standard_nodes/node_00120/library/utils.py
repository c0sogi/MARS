import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Automatically selects the available device (GPU or CPU).

    Returns:
        torch.device: The device object ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
