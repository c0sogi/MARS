import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    logic as defined in the task metric.

    The predicted probabilities are first rescaled so that each row sums to 1.
    Then, they are clipped to the range [1e-15, 1 - 1e-15].
    Finally, the log loss is computed.

    Args:
        y_true (array-like): True labels (can be class indices or one-hot encoded).
        y_pred (array-like): Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays for consistent operations
    y_pred = np.array(y_pred)

    # Rescale probabilities to sum to 1 per row
    # We use keepdims=True to allow broadcasting
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Avoid division by zero in the unlikely event a row sums to 0
    # (though model outputs should generally be positive)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # Clip probabilities to avoid extremes of the log function
    # Range: [10^-15, 1 - 10^-15]
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # Calculate and return the log loss
    # sklearn.metrics.log_loss handles both label encoding and one-hot encoding for y_true
    score = log_loss(y_true, y_pred_clipped)

    return score
