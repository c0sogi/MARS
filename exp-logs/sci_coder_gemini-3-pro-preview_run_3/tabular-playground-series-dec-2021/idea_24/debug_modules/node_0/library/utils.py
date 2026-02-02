import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one A100

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Selects the available hardware device. Prioritizes CUDA (NVIDIA A100) if available.

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
