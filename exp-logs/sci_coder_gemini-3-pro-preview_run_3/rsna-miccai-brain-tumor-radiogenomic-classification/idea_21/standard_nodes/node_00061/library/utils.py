import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value.
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
    Determines the available computational device.

    Returns:
        torch.device: The 'cuda' device if available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
