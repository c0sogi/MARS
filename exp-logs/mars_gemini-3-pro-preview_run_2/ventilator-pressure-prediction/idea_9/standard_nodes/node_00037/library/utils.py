import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device based on the configuration.
    """
    return torch.device(Config.DEVICE)


class MetricMonitor:
    """
    A utility class to track and average metrics (e.g., Loss, MAE) over batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'loss').
            val (float): The value of the metric for the current batch.
            n (int): The number of items in the batch (default: 1).
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}

        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """Returns the average value of a specific metric."""
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """
        Returns a string representation of all tracked metrics with full precision.
        Format: "metric1: value1 | metric2: value2"
        """
        return " | ".join([f"{k}: {self.get_avg(k)}" for k in self.metrics])
