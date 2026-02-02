import os
import random
import numpy as np
import torch
import library.config as config


def seed_everything(seed: int = None):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device (CUDA or CPU) based on availability.

    Returns:
        torch.device: The available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
