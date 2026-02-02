import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configure PyTorch for deterministic execution
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clipped_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss with specific normalization and clipping
    as defined in the task metric.

    The metric rescales rows to sum to 1 and clips probabilities to [1e-15, 1-1e-15].

    Args:
        y_true (array-like): Ground truth labels. Can be a 1D array of labels
                             or a 2D array of one-hot encoded vectors.
        y_pred (array-like): Predicted probabilities. Shape (n_samples, n_classes).
                             Values do not need to sum to 1.

    Returns:
        float: The calculated multi-class log loss.
    """
    # Ensure y_pred is a numpy array for operations
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale prior to being scored (each row is divided by the row sum)
    # We add a tiny epsilon to the sum to prevent division by zero if a row is strictly 0
    row_sums = y_pred.sum(axis=1, keepdims=True)
    y_pred_norm = y_pred / np.maximum(row_sums, 1e-15)

    # 2. Replace predicted probabilities with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Multi-class Log Loss
    # We use sklearn's implementation which handles various y_true formats
    # Note: sklearn also has an internal eps, but our manual clipping ensures
    # strict adherence to the competition's specific bounds before passing it.
    return log_loss(y_true, y_pred_clipped)
