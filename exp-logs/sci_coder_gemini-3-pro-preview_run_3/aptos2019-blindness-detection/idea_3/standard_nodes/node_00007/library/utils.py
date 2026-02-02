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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    This function computes the agreement between two ratings. It is robust to
    regression outputs (floats) by rounding them to the nearest integer before
    calculation, as the metric is defined for ordinal/categorical ratings.

    Args:
        y_true (array-like): Ground truth labels (integers).
        y_pred (array-like): Predicted scores. Can be integers or floats.

    Returns:
        float: The quadratic weighted kappa score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If predictions are floats (from regression), round them to nearest integer
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(y_pred).astype(int)

    # Calculate Cohen's Kappa with quadratic weights
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
