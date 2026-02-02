import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python's random, numpy, torch, and os environments.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clip_log_loss(y_true, y_pred, classes=None):
    """
    Calculates the multi-class log loss with specific rescaling and clipping rules.

    Rules:
    1. Rescale: Each row of probabilities is divided by the row sum.
    2. Clip: Probabilities are clipped to the range [1e-15, 1 - 1e-15].

    Args:
        y_true (array-like): True class labels (1D array of indices or strings).
        y_pred (array-like): Predicted probabilities (2D array, shape [n_samples, n_classes]).
        classes (array-like, optional): List of all class labels. Essential if y_true
                                        in a specific batch does not contain all classes.

    Returns:
        float: The calculated multi-class log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale probabilities: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Handle cases where sum is 0 to avoid division by zero (though unlikely with valid model outputs)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # Pass 'labels' to ensure correct mapping between columns and classes
    score = log_loss(y_true, y_pred_clipped, labels=classes)

    return score


def score_predictions(y_true, y_pred, classes=None):
    """
    Wrapper function to calculate the competition metric (Log Loss).

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted probabilities.
        classes (array-like, optional): List of all unique class labels.

    Returns:
        float: The log loss score.
    """
    return clip_log_loss(y_true, y_pred, classes=classes)
