import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and os.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    rules as defined in the competition metric.

    The submitted probabilities are rescaled prior to being scored (each row is
    divided by the row sum). Predicted probabilities are replaced with
    max(min(p, 1-10^-15), 10^-15).

    Args:
        y_true (array-like): Ground truth labels (1D array of class indices or labels).
        y_pred (array-like): Predicted probabilities (2D array, [n_samples, n_classes]).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale: each row is divided by the row sum
    # We add a tiny epsilon to the sum to strictly avoid division by zero if a row is all zeros
    row_sums = y_pred.sum(axis=1)
    # If row_sum is 0, we leave it (though it will result in NaNs if we divide,
    # but valid probability predictions shouldn't be all zero).
    # To be safe against pure zero rows:
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    # Note: sklearn log_loss does this internally via its 'eps' parameter,
    # but we apply it explicitly here to match the description exactly before passing.
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles the log calculation and averaging.
    # It automatically handles y_true as labels (e.g., strings or integers).
    score = log_loss(y_true, y_pred_clipped, labels=np.unique(y_true))

    return score
