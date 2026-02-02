import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and the PYTHONHASHSEED environment variable
    to ensure reproducible results.

    Args:
        seed (int): The random seed to set. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clipped_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss metric with specific rescaling and clipping
    as defined in the competition task.

    Steps:
    1. Cast predictions to float64.
    2. Rescale rows of y_pred to sum to 1.
    3. Clip probabilities to [1e-15, 1-1e-15].
    4. Compute log loss.

    Args:
        y_true (array-like): Ground truth labels. Can be class indices or one-hot encoded.
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The computed log loss.
    """
    # Ensure high precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale rows to sum to 1 (Metric requirement)
    # We add a safety check for row_sums to avoid division by zero
    row_sums = y_pred.sum(axis=1)

    # If row sum is 0, we cannot normalize. We leave it as is (zeros),
    # which will be handled by clipping (becoming 1e-15).
    mask = row_sums > 0
    y_pred[mask] = y_pred[mask] / row_sums[mask, np.newaxis]

    # Clip probabilities to avoid log(0) extremes (Metric requirement)
    # Range: [1e-15, 1 - 1e-15]
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute Log Loss
    # We rely on sklearn's implementation for the summation and averaging logic.
    # Note: If y_true are integer labels, sklearn assumes they correspond to indices
    # of y_pred columns in sorted order.
    return log_loss(y_true, y_pred)
