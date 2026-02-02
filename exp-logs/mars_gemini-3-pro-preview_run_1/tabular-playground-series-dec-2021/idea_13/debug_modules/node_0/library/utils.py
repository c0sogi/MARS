import os
import random
import numpy as np
import torch


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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically detects and assigns the device (GPU if available, else CPU).

    Returns:
        torch.device: The computing device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def ensure_directory_exists(path: str):
    """
    Ensures that the specified directory exists.

    Args:
        path (str): The directory path to check/create.
    """
    os.makedirs(path, exist_ok=True)


def get_cache_dir() -> str:
    """
    Returns the path to the working directory for this experiment (idea_13),
    ensuring it exists.

    Returns:
        str: The path to the cache directory.
    """
    cache_dir = "./working/idea_13/"
    ensure_directory_exists(cache_dir)
    return cache_dir
