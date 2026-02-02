import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the Multi-class Logarithmic Loss with specific rescaling and clipping
    as defined in the task metric.

    The submitted probabilities are rescaled prior to being scored (each row is
    divided by the row sum). Predicted probabilities are replaced with
    max(min(p, 1-10^-15), 10^-15).

    Args:
        y_true: Ground truth labels. Can be an array of labels or indices.
        y_pred: Predicted probabilities. Shape (n_samples, n_classes).
        labels: Optional list of labels to index the classes.

    Returns:
        float: The calculated log loss value.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid NaN (though unlikely in valid probability outputs)
    # If a row sums to 0, we leave it as 0 (or handle as uniform), but division by 1 preserves it for inspection/error
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of the log function
    # max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    return log_loss(y_true, y_pred, labels=labels)


def ensure_directory(path):
    """
    Ensures that the directory for the given path exists.

    Args:
        path (str): The directory path to check/create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
