import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    # For multi-GPU, though we assume single GPU here
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (like Loss and AUC) during training/validation.
    """

    def __init__(self):
        """
        Initialize the MetricMonitor.
        """
        self.reset()

    def reset(self):
        """
        Resets the internal state of the monitor.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss', 'AUC').
            val (float): The value to record.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without rounding as per requirements.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for metric_name, metric in self.metrics.items()
            ]
        )

    def get_metrics(self):
        """
        Returns a dictionary of the current average metrics.
        """
        return {
            metric_name: metric["avg"] for metric_name, metric in self.metrics.items()
        }
