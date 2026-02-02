import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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
    Returns the appropriate PyTorch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricMonitor:
    """
    A utility class to track and smooth training metrics (Loss, Accuracy, AUC, etc.)
    during epoch iterations.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the metric stores.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name: str, val: float):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss').
            val (float): The current value of the metric.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without rounding.
        """
        return " | ".join(
            [
                "{}: {:.15f}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )

    def get_metrics(self):
        """
        Returns a dictionary of the current average metrics.
        """
        return {k: v["avg"] for k, v in self.metrics.items()}
