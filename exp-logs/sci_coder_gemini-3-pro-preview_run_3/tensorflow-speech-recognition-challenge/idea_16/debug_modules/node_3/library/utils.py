import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricMonitor:
    """
    A helper class to track and average metrics (loss, accuracy, etc.)
    over the course of an epoch.
    """

    def __init__(self, float_precision=4):
        """
        Args:
            float_precision (int): Number of decimal places for string representation.
        """
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Update the running average for a specific metric.

        Args:
            metric_name (str): The name of the metric (e.g., 'loss', 'acc').
            val (float or torch.Tensor): The value to record.
            n (int): The number of samples this value represents (usually batch size).
        """
        if isinstance(val, torch.Tensor):
            val = val.item()

        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}

        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """
        Returns the current average for the specified metric.

        Args:
            metric_name (str): The name of the metric.

        Returns:
            float: The average value, or 0.0 if not found.
        """
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """
        Returns a formatted string containing the averages of all tracked metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    name, self.get_avg(name), prec=self.float_precision
                )
                for name in sorted(self.metrics.keys())
            ]
        )
