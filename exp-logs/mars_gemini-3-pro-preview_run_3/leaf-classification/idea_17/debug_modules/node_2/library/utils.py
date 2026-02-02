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
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log loss extremes.

    Args:
        probs (np.ndarray): Array of probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1.0 - epsilon)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss metric as defined in the task.
    Performs row-wise normalization (rescaling) followed by clipping before scoring.

    Args:
        y_true (array-like): Ground truth labels (strings or indices).
        y_pred (array-like): Predicted probabilities (shape: [n_samples, n_classes]).
        labels (array-like, optional): List of class labels to index the columns of y_pred.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale: each row is divided by the row sum
    # Handle division by zero if sum is 0 (unlikely for valid probs but good for safety)
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero by replacing 0 sums with 1 (result remains 0)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Calculate Log Loss
    # We pass eps=1e-15 to match our clipping, though we already clipped.
    score = log_loss(y_true, y_pred_clipped, labels=labels)

    return score
