import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def clip_probabilities(probs, clip_min=1e-15, clip_max=1.0 - 1e-15):
    """
    Clips probabilities to the range [clip_min, clip_max] to avoid log(0) errors
    and satisfy the metric requirements.

    Args:
        probs (np.ndarray): The array of probabilities.
        clip_min (float): The minimum allowed probability.
        clip_max (float): The maximum allowed probability.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    return np.clip(probs, clip_min, clip_max)


def calculate_log_loss(y_true, y_pred, normalize=True):
    """
    Calculates the multi-class log loss metric with specific preprocessing
    as defined in the task description (normalization and clipping).

    Args:
        y_true (array-like): Ground truth labels (class indices or one-hot).
        y_pred (array-like): Predicted probabilities.
        normalize (bool): Whether to normalize rows to sum to 1 before scoring.
                          Defaults to True as per competition rules.

    Returns:
        float: The calculated log loss.
    """
    # Ensure input is numpy array
    y_pred = np.array(y_pred)

    if normalize:
        # Rescale prior to being scored (each row is divided by the row sum)
        row_sums = y_pred.sum(axis=1)
        # Handle potential zero sums to avoid division by zero (though unlikely for proper probs)
        row_sums[row_sums == 0] = 1.0
        y_pred = y_pred / row_sums[:, np.newaxis]

    # Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
    y_pred = clip_probabilities(y_pred)

    # Calculate log loss
    # Note: sklearn log_loss handles the log calculation and averaging
    score = log_loss(y_true, y_pred)

    return score
