import torch
import random
import numpy as np
from library.config import set_seed as _lib_set_seed
from library.config import ensure_dir as _lib_ensure_dir
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Wraps the implementation from library.config and adds python's random.seed
    to ensure full determinism.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Set python random seed (not covered in library.config)
    random.seed(seed)
    # Set numpy and torch seeds via library.config
    _lib_set_seed(seed)


def get_device() -> torch.device:
    """
    Selects the appropriate device for training or inference.

    Returns:
        torch.device: Returns a CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.
    Wraps library.config.ensure_dir for consistent usage.

    Args:
        path (str): The directory path to check or create.
    """
    _lib_ensure_dir(path)
