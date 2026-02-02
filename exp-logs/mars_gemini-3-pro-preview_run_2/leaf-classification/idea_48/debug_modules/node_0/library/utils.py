import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import FLOAT_PRECISION


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, Numpy, and OS environment variables
    to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def clipped_log_loss(y_true, y_pred, **kwargs):
    """
    Calculates the multi-class log loss metric with specific clipping and rescaling
    logic as defined in the competition rules.

    Logic:
    1. Rescale: Each row of probabilities is divided by its sum to ensure it sums to 1.
    2. Clip: Probabilities are replaced with max(min(p, 1-10^-15), 10^-15).
    3. Score: Calculate standard log loss.

    Args:
        y_true: Ground truth labels (array-like). Can be class indices or label strings.
        y_pred: Predicted probabilities (array-like of shape (n_samples, n_classes)).
        **kwargs: Additional arguments passed to sklearn.metrics.log_loss
                  (e.g., 'labels' to define the full set of class labels if y_true
                  is a subset).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are numpy array with the correct precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)

    # 1. Rescale rows to sum to 1
    # Calculate row sums
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential division by zero if a row sums to 0 (assign 1.0 to avoid NaN,
    # though 0 sum implies all probs are 0 which will be handled by clipping)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip probabilities to avoid extremes of the log function
    # Range: [1e-15, 1 - 1e-15]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Calculate log loss using sklearn
    # We pass the explicitly scaled and clipped probabilities.
    return log_loss(y_true, y_pred, **kwargs)
