import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set python hash seed for consistent hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_roc_auc(y_true, y_scores):
    """
    Computes the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (np.ndarray or list): Ground truth binary labels.
        y_scores (np.ndarray or list): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true
               (handling edge cases in small batches).
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Check if there are at least two classes to calculate ROC AUC
    # Scikit-learn raises an error if only one class is present.
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


def get_device():
    """
    Returns the PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
