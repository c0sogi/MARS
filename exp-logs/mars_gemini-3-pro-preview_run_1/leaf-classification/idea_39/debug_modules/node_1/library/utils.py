import os
import random
import hashlib
import numpy as np
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and hash seed.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    Computes the multi-class log loss with specific rescaling and clipping
    as required by the competition metric.

    The submitted probabilities are not required to sum to one because they are
    rescaled prior to being scored (each row is divided by the row sum).
    Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15).

    Args:
        y_true: Array-like of shape (n_samples,) containing true class labels (integers)
                or (n_samples, n_classes) one-hot encoded.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.
        eps: float, clipping epsilon. Default is 1e-15.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array with float precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide by row sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero; if sum is 0, result remains 0 (handled by clipping later)
    y_pred = np.divide(y_pred, row_sums, out=np.zeros_like(y_pred), where=row_sums != 0)

    # 2. Clip: Replace probabilities with max(min(p, 1-eps), eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Compute Log Loss
    # If y_true is 1D (class indices), sklearn infers labels from data.
    # We explicitly provide labels to ensure all classes are accounted for,
    # assuming y_pred columns correspond to classes 0..K-1.
    labels = None
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        labels = np.arange(y_pred.shape[1])

    return log_loss(y_true, y_pred, labels=labels)


def generate_config_hash(config):
    """
    Generates an MD5 hash from a configuration object (dict, list, etc.)
    to ensure deterministic caching keys.

    Args:
        config: Configuration object (usually a dictionary).

    Returns:
        str: MD5 hash hex string.
    """
    if isinstance(config, dict):
        # Sort by key to ensure consistent string representation regardless of insertion order
        config_str = str(sorted(config.items()))
    else:
        config_str = str(config)

    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
