import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric with explicit row normalization and clipping.

    The metric requires:
    1. Rescaling each row of probabilities to sum to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].
    3. Calculating negative log likelihood.

    Args:
        y_true (array-like): True labels (1D) or one-hot encoded labels (2D).
        y_pred (array-like): Predicted probabilities (2D).

    Returns:
        float: The calculated log loss.
    """
    # Convert predictions to float64 numpy array for precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale rows to sum to 1
    # Calculate row sums
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid division by zero (though unlikely in valid predictions)
    row_sums[row_sums == 0] = 1.0
    # Broadcast division
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles the summation and averaging
    return log_loss(y_true, y_pred)
