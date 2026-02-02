import os
import random
import numpy as np
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and python hash.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clipped_log_loss(y_true, y_pred, **kwargs):
    """
    Computes the multi-class log loss with specific clipping and normalization rules
    as defined in the competition metric.

    Rules:
    1. Rescale probabilities so each row sums to 1.
    2. Clip probabilities to [1e-15, 1 - 1e-15].
    3. Compute Multi-class log loss.

    Args:
        y_true (array-like): Ground truth (correct) labels for n_samples samples.
        y_pred (array-like): Predicted probabilities, as returned by a classifier’s
                             predict_proba method. Shape = [n_samples, n_classes].
        **kwargs: Additional keyword arguments passed to sklearn.metrics.log_loss
                  (e.g., 'labels').

    Returns:
        float: The calculated log loss.
    """
    # Ensure prediction is a numpy array for element-wise operations
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1
    # Calculate row sums, keeping dimensions for broadcasting
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Handle potential zero sum rows by replacing sum with 1.0 to avoid division by zero
    # (Though in valid probability predictions this shouldn't happen, it's a safety check)
    row_sums[row_sums == 0] = 1.0

    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    # The task specifies: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Compute Log Loss
    # We delegate the actual loss calculation to sklearn, which handles label encoding/matching.
    return log_loss(y_true, y_pred_clipped, **kwargs)
