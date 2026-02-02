import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate device (cuda or cpu) for torch operations.

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
