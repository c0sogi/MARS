import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = Config.DEVICE) -> torch.device:
    """
    Returns the available device (CUDA or CPU).

    Args:
        device_name (str): The name of the device to use (e.g., 'cuda', 'cpu').
                           Defaults to Config.DEVICE.

    Returns:
        torch.device: The PyTorch device object.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        # Fallback to CPU if CUDA is requested but not available
        return torch.device("cpu")

    return torch.device(device_name)


def print_metric(name: str, value: float) -> None:
    """
    Prints a metric with full precision as required.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
