import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    This function adapts continuous regression predictions to the ordinal
    integer scale required by the metric.

    Args:
        y_true (array-like): Ground truth labels (0-4).
        y_pred (array-like): Predicted scores (continuous or integer).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert regression outputs to integer labels
    # Round to nearest integer and clip to valid range [0, 4]
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_clipped = np.clip(y_pred_rounded, 0, 4)

    # Ensure ground truth is integer
    y_true_int = y_true.astype(int)

    return cohen_kappa_score(y_true_int, y_pred_clipped, weights="quadratic")
