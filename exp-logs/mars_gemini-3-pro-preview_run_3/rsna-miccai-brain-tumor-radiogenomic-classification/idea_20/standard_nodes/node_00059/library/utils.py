import os
import random
import numpy as np
import torch
import sys


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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_message(message: str):
    """
    Prints a message to stdout and flushes the buffer to ensure immediate logging.

    Args:
        message (str): The message to log.
    """
    print(message)
    sys.stdout.flush()


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric (e.g., "Validation AUC").
        value (float): The numerical value of the metric.
    """
    # Using default string formatting for float preserves precision better than f"{v:.4f}"
    log_message(f"{name}: {value}")


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
