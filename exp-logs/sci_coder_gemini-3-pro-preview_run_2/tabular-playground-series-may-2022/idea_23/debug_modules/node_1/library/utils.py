import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Ensure inputs are numpy arrays for safety
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate AUC
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle edge cases where only one class is present in y_true
        score = 0.5

    return float(score)
