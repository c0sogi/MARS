import random
import os
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed=Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and print running averages of metrics (e.g., Loss, Accuracy, ROC-AUC).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the monitor."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the running average for a given metric.

        Args:
            metric_name (str): The name of the metric (e.g., "Loss").
            val (float or torch.Tensor): The value to add.
        """
        metric = self.metrics[metric_name]

        # Handle PyTorch tensors by extracting the scalar value
        if isinstance(val, torch.Tensor):
            val = val.item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current metric averages.
        Prints full precision without rounding.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )
