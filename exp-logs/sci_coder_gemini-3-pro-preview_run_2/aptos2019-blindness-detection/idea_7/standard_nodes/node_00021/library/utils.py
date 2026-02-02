import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) metric between ground truth and predictions.

    This function handles regression outputs by rounding them to the nearest integer
    before calculating the kappa score, which is appropriate for the ordinal nature
    of the diabetic retinopathy labels (0-4).

    Args:
        y_true (array-like): Ground truth labels (integers).
        y_pred (array-like): Predicted scores (can be floats from regression or integers).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Ensure y_true are integers
    y_true = np.asarray(y_true, dtype=int)

    # Round predicted scores to the nearest integer and convert to int
    # This handles the output of the regression model
    y_pred = np.rint(np.asarray(y_pred)).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The number of samples associated with this value (weight).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
