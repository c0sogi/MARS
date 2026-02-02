import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The QWK score.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class MetricMonitor:
    """
    A utility class to track and update metrics (e.g., Loss, Accuracy) during training.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Current value of the metric.
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
        Returns a formatted string of current average metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for metric_name, metric in self.metrics.items()
            ]
        )
