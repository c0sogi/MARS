import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates).

    Returns:
        float: The computed AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if there is only one class in the batch
        return 0.5


class MetricMonitor:
    """
    A utility class to track metrics (loss, accuracy, etc.) during training.
    It maintains a running average of the metrics.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal state of the monitor."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, count=1):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add.
            count (int): The number of samples associated with this value (default 1).
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * count
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """Returns the current average for a specific metric."""
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """Returns a string representation of the current averages."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
