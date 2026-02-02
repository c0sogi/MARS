import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed: int = config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for training and inference.

    Returns:
        torch.device: The computed device (cuda or cpu).
    """
    # config.DEVICE is already determined as 'cuda' or 'cpu'
    return torch.device(config.DEVICE)
