import sys
import os
from sklearn.metrics import roc_auc_score
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation provided in library.config.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC) between predicted probabilities and observed targets.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: Area Under the ROC Curve.
    """
    return roc_auc_score(y_true, y_pred)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all internal statistics to zero.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add (e.g., current batch loss).
            n (int): The number of samples associated with this value (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
