import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels (can be continuous).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Handle torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Since the metric relies on a confusion matrix of ratings 0-4,
    # we must round continuous predictions to the nearest integer
    # and clip them to the valid range.
    y_pred_int = np.rint(y_pred).clip(0, 4).astype(int)
    y_true_int = y_true.astype(int)

    return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")


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
