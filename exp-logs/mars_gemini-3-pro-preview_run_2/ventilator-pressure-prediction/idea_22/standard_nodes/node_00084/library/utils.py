import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility across
    Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and update metrics (e.g., loss, MAE) during
    training and validation loops.
    """

    def __init__(self, float_precision=4):
        """
        Args:
            float_precision (int): The number of decimal places to use in the
                                   string representation (for progress bars).
        """
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics to zero.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates a specific metric with a new value.

        Args:
            metric_name (str): The name of the metric (e.g., "Loss").
            val (float): The value to update (e.g., current batch loss).
            n (int): The number of items in the batch (weight for the average).
        """
        metric = self.metrics[metric_name]

        # Accumulate the weighted sum
        metric["val"] += val * n
        metric["count"] += n
        # Recalculate the running average
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a formatted string representation of the tracked metrics.
        Useful for printing progress bars.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(name, metric["avg"], prec=self.float_precision)
                for name, metric in self.metrics.items()
            ]
        )

    def get_avg(self, metric_name):
        """
        Returns the current average value of a specific metric with full precision.
        """
        return self.metrics[metric_name]["avg"]

    def get_all_metrics(self):
        """
        Returns a dictionary of all current average metrics with full precision.
        Useful for logging final epoch results without rounding.
        """
        return {name: metric["avg"] for name, metric in self.metrics.items()}
