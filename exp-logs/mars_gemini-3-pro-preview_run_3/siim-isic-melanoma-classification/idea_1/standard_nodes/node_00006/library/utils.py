import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (like Loss) during training
    or validation loops. It maintains a running average of the metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the internal metric storage.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss').
            val (float): The value to add (typically the mean value for the batch).
            n (int): Weight of the value (typically the batch size).
        """
        metric = self.metrics[metric_name]

        # val is usually the mean loss of the batch, so we multiply by n to get the sum
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """
        Returns the current average for a specific metric.
        """
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without rounding.
        """
        # Using str() on the float ensures full precision is kept in the string representation
        return " | ".join(
            ["{}: {}".format(k, v["avg"]) for k, v in self.metrics.items()]
        )
