import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (np.ndarray or list): Ground truth labels or probabilities.
                                     Shape: (n_samples, n_classes).
        y_pred (np.ndarray or list): Predicted probabilities.
                                     Shape: (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # sklearn.metrics.log_loss handles epsilon clipping automatically (eps=1e-15 by default)
    # It supports both class labels and soft probability targets.
    return log_loss(y_true, y_pred)
