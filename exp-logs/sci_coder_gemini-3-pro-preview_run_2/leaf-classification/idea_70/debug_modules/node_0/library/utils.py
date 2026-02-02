import os
import random
import numpy as np
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch (if available).

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with row-normalization and specific clipping
    as defined in the task metric.

    The probabilities are rescaled such that each row sums to 1.
    Then, probabilities are clipped to [1e-15, 1 - 1e-15].

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are float64 for precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid NaN (though unlikely in valid predictions)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of the log function
    # Formula: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate log loss
    return log_loss(y_true, y_pred)
