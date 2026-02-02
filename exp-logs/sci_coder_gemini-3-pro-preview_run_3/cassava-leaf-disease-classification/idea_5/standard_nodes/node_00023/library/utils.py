import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one here

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track metrics (like Loss and Accuracy) during training/validation.
    Calculates and stores the running average of the metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the metrics state."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a given metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to record (usually the average for the batch).
            n (int): The number of samples corresponding to val (usually batch size).
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current metric averages.
        Prints full precision as requested.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )
