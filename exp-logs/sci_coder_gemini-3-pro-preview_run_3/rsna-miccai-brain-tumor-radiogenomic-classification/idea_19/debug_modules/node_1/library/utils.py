import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
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


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_metrics(metrics_dict):
    """
    Prints validation metrics with full precision.

    Args:
        metrics_dict (dict): A dictionary where keys are metric names and values are metric scores.
    """
    # Format string to print key-value pairs without rounding
    message_parts = [f"{k}: {v}" for k, v in metrics_dict.items()]
    print(" | ".join(message_parts))
