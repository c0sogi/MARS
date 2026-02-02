import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Returns the device available for computation.

    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
