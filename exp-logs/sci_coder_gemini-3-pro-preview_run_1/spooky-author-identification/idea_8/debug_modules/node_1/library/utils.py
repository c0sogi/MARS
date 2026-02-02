import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss with specific clipping and normalization
    as required by the competition metric.

    The predicted probabilities are rescaled (each row divided by row sum) and then
    clipped to the range [1e-15, 1 - 1e-15] before scoring.

    Args:
        y_true: Ground truth labels. Can be an array of integers (class indices)
                or a 2D array of one-hot encoded labels.
        y_pred: Predicted probabilities. A 2D array of shape (n_samples, n_classes).
                Can be a numpy array, torch tensor, or list.

    Returns:
        float: The calculated log loss.
    """
    # Convert tensors/lists to numpy if necessary
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()
    elif hasattr(y_pred, "numpy") and not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    elif hasattr(y_true, "numpy") and not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)

    # Ensure y_pred is float for division
    y_pred = y_pred.astype(np.float64)

    # 1. Rescale: each row is divided by the row sum
    # This ensures probabilities sum to 1, as per competition metric requirements.
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero; if sum is 0, we set divisor to 1 (values remain 0)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    # This prevents infinite loss values for 0 or 1 probabilities.
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles integer labels or one-hot encoding automatically.
    score = log_loss(y_true, y_pred)

    return score
