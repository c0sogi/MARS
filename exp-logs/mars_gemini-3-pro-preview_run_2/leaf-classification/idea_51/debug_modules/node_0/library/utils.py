import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, PROB_CLIP


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def calculate_log_loss(y_true, y_pred, eps=PROB_CLIP):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    logic as defined in the task metric.

    The metric requires:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to [eps, 1-eps].
    3. Calculating the negative log likelihood.

    Args:
        y_true (array-like): True labels. Can be 1D array of class indices/names
                             or 2D indicator array.
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).
        eps (float): Epsilon for clipping. Defaults to PROB_CLIP (1e-15).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are a numpy array
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide each row by its sum
    # We add a tiny constant to row_sums to avoid division by zero if a model outputs all zeros
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums safely
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Replace probabilities with max(min(p, 1-eps), eps)
    # Note: sklearn.metrics.log_loss handles clipping via the 'eps' argument,
    # but we perform it here explicitly to ensure the exact values passed to the metric
    # match the competition's preprocessing description.
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Calculate Log Loss
    # We rely on sklearn's implementation to handle label matching and calculation
    return log_loss(y_true, y_pred, eps=eps)
