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
    A utility class to track and calculate running averages of metrics (e.g., Loss, F1-Score).
    Designed to print metrics with full precision as per requirements.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the monitor."""
        self.val = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss').
            val (float): The value of the metric observed.
            n (int): The number of samples associated with this value (usually batch size).
        """
        self.val[metric_name] += val * n
        self.count[metric_name] += n
        self.avg[metric_name] = self.val[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the metrics.
        Prints full precision without rounding or formatting.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, self.avg[metric_name])
                for metric_name in self.avg
            ]
        )
