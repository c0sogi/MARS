import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure
    reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one A100

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines the computation device based on CUDA availability.

    Returns:
        torch.device: Device object ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
