import os
import random
import numpy as np
import torch
from library.config import SEED, DEVICE


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures cuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device():
    """
    Returns the PyTorch device to be used based on configuration and availability.

    Returns:
        torch.device: The device object (cpu or cuda).
    """
    # logic to fallback if config says cuda but unavailable
    d = DEVICE
    if d == "cuda" and not torch.cuda.is_available():
        d = "cpu"

    return torch.device(d)
