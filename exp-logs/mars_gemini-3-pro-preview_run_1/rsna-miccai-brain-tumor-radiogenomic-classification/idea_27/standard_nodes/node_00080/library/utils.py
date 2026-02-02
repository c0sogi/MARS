import os
import random
import sys
import numpy as np
import torch
from library.config import SEED, DEVICE


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device to be used for training/inference.

    Returns:
        torch.device: The device object (cpu or cuda).
    """
    return torch.device(DEVICE)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy during training loops.
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


def print_header(title):
    """
    Prints a formatted header block to the console to separate experiment sections.

    Args:
        title (str): The title text to display.
    """
    print(f"\n{'='*40}")
    print(f" {title}")
    print(f"{'='*40}")


def print_metric(name, value):
    """
    Prints a metric with full precision without rounding, as required.

    Args:
        name (str): Name of the metric (e.g., 'Validation AUC').
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
