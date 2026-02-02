import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Automatically selects the available device (GPU or CPU).

    Args:
        use_cuda (bool): If True, attempts to use CUDA if available.
                         If False, forces CPU usage. Default is True.

    Returns:
        torch.device: The device to be used for computation.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
