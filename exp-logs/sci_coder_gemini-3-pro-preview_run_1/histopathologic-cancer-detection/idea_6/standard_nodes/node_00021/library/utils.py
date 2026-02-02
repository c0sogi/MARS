import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config implementation to ensure consistency.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    Config.set_seed(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add (e.g., current batch loss).
            n (int): The weight of the value (e.g., batch size). Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The computed AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Convert inputs to numpy arrays for safety
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge case where y_true contains only one class
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
