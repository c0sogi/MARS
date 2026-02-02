import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from library.config import Config


def seed_everything(seed=None):
    """
    Sets seeds for reproducibility by delegating to the Config class.
    """
    Config.seed_everything(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (1D array-like or tensor).
        y_pred: Predicted probabilities (1D array-like or tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


class MetricMonitor:
    """
    A utility class to track running averages of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics.
        """
        self.val = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to update (e.g., batch loss).
            n (int): Weight of the value (e.g., batch size).
        """
        self.val[metric_name] += val * n
        self.count[metric_name] += n
        self.avg[metric_name] = self.val[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join([f"{k}: {v}" for k, v in self.avg.items()])
