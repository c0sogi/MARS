import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device based on availability.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
