import logging
import sys
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by delegating to the Config class.
    """
    Config.seed_everything(seed)


def get_logger(name="cactus_pipeline"):
    """
    Creates and configures a logger that outputs to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricMonitor:
    """
    A utility class to track and update metrics (loss, accuracy, etc.) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all tracked metrics.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the specified metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current average metrics.
        Prints full precision without rounding as requested.
        """
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (Tensor or numpy array).
        y_pred: Predicted probabilities (Tensor or numpy array).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true only has one class in the current batch.
        # In such cases, ROC AUC is undefined. returning 0.5 is a neutral fallback for logging.
        return 0.5
