import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (e.g., Loss, Accuracy) over an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics to zero.
        """
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val):
        """
        Updates the state of a specific metric with a new value.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss').
            val (float): The current value of the metric to record.
        """
        self.val[metric_name] = val
        self.sum[metric_name] += val
        self.count[metric_name] += 1
        self.avg[metric_name] = self.sum[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics.
        Prints full precision without rounding as per requirements.
        """
        return " | ".join(
            [f"{metric_name}: {self.avg[metric_name]}" for metric_name in self.avg]
        )
