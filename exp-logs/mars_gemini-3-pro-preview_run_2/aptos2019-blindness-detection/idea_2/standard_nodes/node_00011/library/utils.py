import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa score.

    Since the model is a regression model, predictions are continuous.
    This function clips them to the range [0, 4] and rounds them to the
    nearest integer before calculating the metric.

    Args:
        y_true (array-like): Ground truth labels (integers 0-4).
        y_pred (array-like): Predicted scores (continuous).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Process predictions: Clip to valid range and round to nearest integer
    y_pred_clamped = np.clip(y_pred, 0, 4)
    y_pred_rounded = np.round(y_pred_clamped).astype(int)

    # Calculate Quadratic Weighted Kappa
    # weights='quadratic' is required for this specific metric
    score = cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")

    return score
