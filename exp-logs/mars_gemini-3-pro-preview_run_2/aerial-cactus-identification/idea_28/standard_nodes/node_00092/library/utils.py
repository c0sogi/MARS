import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # print(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Determines and returns the available computational device (GPU or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print(f"Device selected: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        # print(f"Device selected: {device}")

    return device
