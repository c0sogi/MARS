import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the torch device based on configuration.
    """
    return torch.device(Config.DEVICE)


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.)
    over a training or validation epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'MAE').
            val (float): The value of the metric for the current batch.
            n (int): The number of samples in the batch (weight).
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
                "{}: {}".format(metric_name, metric["avg"])
                for (metric_name, metric) in self.metrics.items()
            ]
        )
