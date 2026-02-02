import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
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

    print(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Returns the appropriate torch device based on availability.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_metric(phase: str, metric_name: str, value: float):
    """
    Prints a metric value with full precision.

    Args:
        phase (str): The phase of execution (e.g., 'Train', 'Val').
        metric_name (str): The name of the metric (e.g., 'Loss', 'AUC').
        value (float): The metric value.
    """
    print(f"{phase} {metric_name}: {value}")


def print_config():
    """
    Prints the configuration parameters defined in the Config class.
    """
    print("=" * 30)
    print("CONFIGURATION")
    print("=" * 30)
    for key, value in Config.__dict__.items():
        if not key.startswith("__") and not callable(value):
            print(f"{key}: {value}")
    print("=" * 30)
