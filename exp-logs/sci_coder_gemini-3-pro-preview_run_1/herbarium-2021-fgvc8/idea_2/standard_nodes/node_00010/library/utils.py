import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers in random, numpy, and torch.
    Ensures deterministic behavior for reproducibility.

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


class MetricMonitor:
    """
    A utility class to compute and keep track of the running average of metrics.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal state of the metrics."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric.

        Args:
            metric_name (str): The name of the metric.
            val (float): The value to update (e.g., average batch loss).
            n (int): The number of samples associated with the value (usually batch size).
        """
        metric = self.metrics[metric_name]

        # val is typically the mean over the batch, so we multiply by n to get the sum
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """Returns a string representation of the metrics with specified precision."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def calculate_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels.
        y_pred (torch.Tensor or np.ndarray): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return f1_score(y_true, y_pred, average="macro")
