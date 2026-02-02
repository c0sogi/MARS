import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure deterministic behavior.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def get_device() -> torch.device:
    """
    Returns the PyTorch device configured for the environment.

    Returns:
        torch.device: The device (CPU or CUDA).
    """
    return torch.device(Config.DEVICE)


def print_metric(name: str, value: float) -> None:
    """
    Prints a validation metric with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    # Requirement: print the full precision without any rounding or formatting
    print(f"{name}: {value}")


def log_message(message: str) -> None:
    """
    Simple wrapper for logging messages to stdout.

    Args:
        message (str): The message to log.
    """
    print(message)


def ensure_directory(path: str) -> None:
    """
    Ensures that the specified directory exists.

    Args:
        path (str): The directory path.
    """
    os.makedirs(path, exist_ok=True)
