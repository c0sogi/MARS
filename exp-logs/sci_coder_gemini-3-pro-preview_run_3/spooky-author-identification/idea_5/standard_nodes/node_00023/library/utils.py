import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): The name of the metric.
            val (float): The value to update (e.g., batch loss).
            n (int): The number of samples associated with this value (weight).
                     Defaults to 1.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without explicit rounding.
        """
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']}"
                for metric_name, metric in self.metrics.items()
            ]
        )
