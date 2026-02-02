import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The AUC score.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5


class MetricMonitor:
    """
    A utility class to track running averages of metrics (e.g., Loss)
    during training or validation loops.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val):
        """
        Update the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to update.
        """
        previous_data = self.metrics.get(metric_name, {"count": 0, "sum": 0})

        previous_data["count"] += 1
        previous_data["sum"] += val

        self.metrics[metric_name] = previous_data

    def get_avg(self, metric_name):
        """
        Returns the average value of the metric.
        """
        data = self.metrics.get(metric_name)
        if not data or data["count"] == 0:
            return 0.0
        return data["sum"] / data["count"]

    def __str__(self):
        """
        Returns a formatted string of all tracked metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    name, self.get_avg(name), prec=self.float_precision
                )
                for name in self.metrics
            ]
        )
