import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import SEED, FLOAT_PRECISION


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, os, and torch (if available).

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def normalize_probabilities(y_pred):
    """
    Rescales probabilities so that each row sums to 1, as required by the competition metric.

    Args:
        y_pred (np.ndarray): The predicted probabilities.

    Returns:
        np.ndarray: The row-normalized probabilities.
    """
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle rows that sum to zero (though unlikely with proper models) to avoid NaNs
    row_sums[row_sums == 0] = 1.0
    return y_pred / row_sums


def clip_probabilities(y_pred):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid extremes in the log function.

    Args:
        y_pred (np.ndarray): The predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)
    epsilon = 1e-15
    # Formula: max(min(p, 1-10^-15), 10^-15)
    return np.clip(y_pred, epsilon, 1.0 - epsilon)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss.
    Applies the competition-specific normalization and clipping steps before scoring.

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted probabilities.
        labels (array-like, optional): If y_true are labels, this list defines the column order of y_pred.

    Returns:
        float: The log loss score.
    """
    # 1. Rescale rows to sum to 1
    y_pred_norm = normalize_probabilities(y_pred)

    # 2. Clip probabilities
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Calculate Log Loss
    # We use sklearn's implementation which handles label encoding and multiclass logic
    score = log_loss(y_true, y_pred_clipped, labels=labels)

    return score
