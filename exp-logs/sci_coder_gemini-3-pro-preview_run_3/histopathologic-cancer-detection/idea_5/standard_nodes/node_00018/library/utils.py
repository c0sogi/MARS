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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track metrics (loss, accuracy, AUC, etc.) during training and validation.
    Calculates and stores the running average of the metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss', 'AUC').
            val (float): The value to update.
        """
        self.val[metric_name] = val
        self.sum[metric_name] += val
        self.count[metric_name] += 1
        self.avg[metric_name] = self.sum[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without rounding or formatting as required.
        """
        return " | ".join([f"{name}: {self.avg[name]}" for name in self.avg])
