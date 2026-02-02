import os
import random
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
    A utility class to track and calculate running averages of metrics during training.
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
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add (e.g., batch loss).
            n (int): The number of samples associated with this value (default 1).
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """Returns the current average of the specified metric."""
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """Returns a formatted string of all tracked metrics."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the log loss (binary cross-entropy) between true labels and predictions.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate log loss. sklearn handles clipping internally (eps=1e-15)
    # labels=[0, 1] ensures it works even if y_true only contains one class
    return log_loss(y_true, y_pred, labels=[0, 1])
