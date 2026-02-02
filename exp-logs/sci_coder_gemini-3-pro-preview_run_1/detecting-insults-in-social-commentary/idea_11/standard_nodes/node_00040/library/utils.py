import os
import random
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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
    A utility class to track metrics (loss, accuracy, AUC, etc.) during training/validation.
    Computes the running average of the metrics.
    """

    def __init__(self):
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
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]

        # Handle tensor inputs by converting to float
        if torch.is_tensor(val):
            val = val.item()

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """
        Returns the current average for a specific metric.
        """
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, metric["avg"])
                for (metric_name, metric) in self.metrics.items()
            ]
        )
