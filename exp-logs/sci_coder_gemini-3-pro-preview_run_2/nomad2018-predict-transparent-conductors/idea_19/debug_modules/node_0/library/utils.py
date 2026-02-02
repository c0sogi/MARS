import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the value in Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines and returns the PyTorch device to use (CPU or CUDA).

    Returns:
        torch.device: The device object based on availability and Config.
    """
    # Config.DEVICE is already determined in the config file, but we ensure a torch.device object is returned
    return torch.device(Config.DEVICE)
