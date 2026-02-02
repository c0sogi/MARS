import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (np.array or pd.DataFrame): Ground truth labels (one-hot or probabilities).
                                           Shape: (n_samples, n_classes)
        y_pred (np.array or pd.DataFrame): Predicted probabilities.
                                           Shape: (n_samples, n_classes)

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays for consistency
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Sklearn's log_loss handles clipping (eps) internally to avoid log(0)
    # Default eps is 1e-15, which is standard for 'auto'
    score = log_loss(y_true, y_pred)

    return score
