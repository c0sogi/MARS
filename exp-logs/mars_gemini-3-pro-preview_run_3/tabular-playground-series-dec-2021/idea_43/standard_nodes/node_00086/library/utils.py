import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False):
    """
    Sets seeds for all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed.
        deterministic (bool): If True, sets CuDNN to deterministic mode (slower).
                              If False, enables CuDNN benchmark (faster, less reproducible).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Returns the compute device (GPU if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
