import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def save_checkpoint(state, filename):
    """
    Saves the provided state dictionary to a file.
    Ensures the parent directory exists.

    Args:
        state (dict): The state dictionary to save (e.g., model.state_dict()).
        filename (str): The path to the file where the checkpoint will be saved.
    """
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filename)


class MetricMonitor:
    """
    A utility class to track and calculate the running average of metrics
    (e.g., Loss, MAE) during training or validation.
    """

    def __init__(self):
        self.metrics = defaultdict(lambda: {"sum": 0.0, "count": 0, "avg": 0.0})

    def reset(self):
        """Resets all tracked metrics."""
        self.metrics.clear()

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'Loss').
            val (float): The value to update (typically the mean of a batch).
            n (int): The number of samples associated with 'val' (typically batch size).
        """
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n
        self.metrics[metric_name]["avg"] = (
            self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]
        )

    def get_avg(self, metric_name):
        """Returns the current average for the specified metric."""
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of all metrics with full precision.
        Format: "Metric1: Value1 | Metric2: Value2"
        """
        return " | ".join(
            [f"{key}: {self.metrics[key]['avg']}" for key in self.metrics]
        )
