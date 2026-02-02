import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is not installed or not required for this specific run
        pass


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the Multi-class Log Loss with specific clipping and normalization
    rules as defined in the task description.

    Steps:
    1. Rescale rows of y_pred to sum to 1.
    2. Clip probabilities to [1e-15, 1 - 1e-15].
    3. Calculate log loss.

    Args:
        y_true (array-like): True labels (n_samples,). Can be class indices or string labels
                             if the estimator handles them, but typically indices for log_loss.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale prior to scoring (each row is divided by the row sum)
    # Add a small epsilon to denominator to prevent division by zero if a row sums to 0
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    # "predicted probabilities are replaced with max(min(p,1-10^{-15}),10^{-15})"
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn log_loss handles the log calculation and summation
    score = log_loss(y_true, y_pred_clipped)

    return score
