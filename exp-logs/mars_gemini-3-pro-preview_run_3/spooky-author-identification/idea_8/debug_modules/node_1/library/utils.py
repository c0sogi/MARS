import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probas):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log(0) and
    comply with the competition metric specification.

    Formula: max(min(p, 1-10^-15), 10^-15)

    Args:
        probas (np.ndarray): The probability matrix.

    Returns:
        np.ndarray: The clipped probability matrix.
    """
    eps = 1e-15
    return np.clip(probas, eps, 1 - eps)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss.

    This function mimics the competition scoring mechanism:
    1. Rescales rows of y_pred so they sum to 1.
    2. Clips probabilities to [1e-15, 1-1e-15].
    3. Calculates log loss.

    Args:
        y_true (array-like): True labels (indices or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1
    # "The submitted probabilities for a given sentences are not required to sum to one
    # because they are rescaled prior to being scored (each row is divided by the row sum)."
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    y_pred_clipped = clip_probabilities(y_pred_rescaled)

    # 3. Calculate Log Loss
    # We use sklearn's log_loss. We pass the clipped probabilities.
    # eps is set to 1e-15 to match our manual clipping.
    return log_loss(y_true, y_pred_clipped, eps=1e-15)
