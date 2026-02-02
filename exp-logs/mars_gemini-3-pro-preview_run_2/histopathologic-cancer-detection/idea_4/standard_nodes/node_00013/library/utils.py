import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array, list, or torch.Tensor).
        y_pred: Predicted probabilities (numpy array, list, or torch.Tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Detach and convert to numpy if inputs are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert to numpy arrays if lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if there is only one class in y_true
        return 0.5


class MetricMonitor:
    """
    A class to track metrics (like loss and accuracy) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the metrics state.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float or torch.Tensor): Value to add.
            n (int): Weight of the value (e.g., batch size). Defaults to 1.
        """
        metric = self.metrics[metric_name]

        if isinstance(val, torch.Tensor):
            val = val.item()

        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

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
