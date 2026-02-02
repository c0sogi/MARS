import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function ensures that the results are reproducible by fixing the seed
    for the random number generators in random, numpy, and torch. It also
    configures cuDNN to be deterministic.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    # This might impact performance slightly but is necessary for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Selects the available hardware device for PyTorch operations.

    Checks if a CUDA-enabled GPU is available and returns the corresponding
    device. Otherwise, returns the CPU device.

    Returns:
        torch.device: The 'cuda' device if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
