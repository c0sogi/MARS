import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Determines and returns the available PyTorch device.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def print_metric(name: str, value: float) -> None:
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric (e.g., 'Validation LogLoss').
        value (float): The numerical value of the metric.
    """
    print(f"{name}: {value}")
