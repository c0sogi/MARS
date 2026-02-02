import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricMonitor:
    """
    A utility class to track and average metrics (e.g., loss, accuracy)
    during training and validation loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics.
        """
        self.val = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val):
        """
        Updates the specified metric with a new value.

        Args:
            metric_name (str): The name of the metric.
            val (float): The value to add.
        """
        self.val[metric_name] += val
        self.count[metric_name] += 1
        self.avg[metric_name] = self.val[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics.
        Prints full precision without rounding or formatting as required.
        """
        return " | ".join(
            [f"{metric_name}: {self.avg[metric_name]}" for metric_name in self.avg]
        )
