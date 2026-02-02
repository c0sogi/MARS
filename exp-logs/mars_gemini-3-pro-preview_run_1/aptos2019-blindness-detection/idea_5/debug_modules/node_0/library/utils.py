import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int):
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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) between ground truth and predictions.

    Args:
        y_true: Array-like of ground truth labels (integers 0-4).
        y_pred: Array-like of predicted labels (integers 0-4).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integer arrays for Cohen's Kappa
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Calculate Quadratic Weighted Kappa
    # weights='quadratic' ensures the metric penalizes larger discrepancies more heavily
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
