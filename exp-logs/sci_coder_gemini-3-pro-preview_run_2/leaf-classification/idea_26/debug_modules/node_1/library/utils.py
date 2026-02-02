import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for Python's random module, numpy, and the PYTHONHASHSEED environment variable
    to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def clipped_log_loss(y_true, y_pred, eps=1e-15, **kwargs):
    """
    Computes the multi-class log loss with specific rescaling and clipping rules.

    The submitted probabilities are rescaled prior to being scored (each row is divided by the row sum).
    Predicted probabilities are replaced with max(min(p, 1-eps), eps).

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted probabilities, shape (n_samples, n_classes).
        eps (float): Clipping epsilon. Defaults to 1e-15.
        **kwargs: Additional arguments passed to sklearn.metrics.log_loss (e.g., labels).

    Returns:
        float: The calculated log loss.
    """
    y_pred = np.array(y_pred)

    # 1. Rescale: Divide each row by the row sum
    # We add a tiny constant to the denominator to prevent division by zero if a model outputs all zeros
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred_norm = y_pred / (row_sums + 1e-15)

    # 2. Clip: Restrict probabilities to [eps, 1-eps]
    # Note: sklearn's log_loss does this internally if eps is provided,
    # but we do it explicitly here to ensure the normalized values are the ones being clipped
    # and strictly follow the metric definition.
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Compute Log Loss
    return log_loss(y_true, y_pred_clipped, **kwargs)
