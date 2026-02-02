import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
    Returns the PyTorch device based on availability and Config.

    Returns:
        torch.device: The device to use for computation.
    """
    return torch.device(Config.DEVICE)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Prints metrics with full precision without rounding.

    Args:
        metrics (dict): Dictionary containing metric names and values.
        prefix (str): Optional prefix for the print statement (e.g., "Validation").
    """
    message_parts = []
    if prefix:
        message_parts.append(f"[{prefix}]")

    for k, v in metrics.items():
        # Using str(v) or default formatting to avoid explicit rounding
        message_parts.append(f"{k}: {v}")

    print(" ".join(message_parts))
