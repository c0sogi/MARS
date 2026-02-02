import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
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
    Calculates the Multi Class Log Loss.

    Args:
        y_true (array-like): Ground truth labels. Can be a 1D array of class indices/labels.
        y_pred (array-like): Predicted probabilities. A 2D array of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # sklearn.metrics.log_loss handles multiclass classification automatically
    # when y_pred is a matrix of probabilities.
    # Explicitly provide labels to handle cases where y_true doesn't cover all classes (e.g. Debug mode)
    labels = list(range(y_pred.shape[1]))
    return log_loss(y_true, y_pred, labels=labels)
