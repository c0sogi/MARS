import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines the available computational device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_metric(name: str, value: float) -> None:
    """
    Prints a metric name and its value with full precision, avoiding rounding.

    Args:
        name (str): The name of the metric (e.g., 'Validation AUC').
        value (float): The numerical value of the metric.
    """
    print(f"{name}: {value}")
