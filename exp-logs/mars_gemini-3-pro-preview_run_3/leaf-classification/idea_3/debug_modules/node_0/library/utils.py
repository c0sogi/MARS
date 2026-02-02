import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [10^-15, 1 - 10^-15] to satisfy the
    metric requirements and avoid extremes of the log function.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1 - epsilon)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss according to the competition metric.

    The metric specifies that submitted probabilities are:
    1. Rescaled so each row sums to 1.
    2. Clipped to [10^-15, 1-10^-15].
    3. Scored using multi-class log loss.

    Args:
        y_true (np.ndarray or list): Ground truth labels (class indices or one-hot).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1
    # We add a tiny epsilon to the sum to prevent division by zero if a row is all zeros
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles both label indices and one-hot encoding for y_true
    return log_loss(y_true, y_pred_clipped)
