import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed to ensure consistent hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth labels (0 or 1). Can be list, numpy array, or torch Tensor.
        y_pred (array-like): Predicted probabilities. Can be list, numpy array, or torch Tensor.

    Returns:
        float: The ROC AUC score. Returns 0.5 if calculation fails due to single-class input.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        # Check if we have both classes to avoid ValueError from roc_auc_score
        if len(np.unique(y_true)) < 2:
            return 0.5

        score = roc_auc_score(y_true, y_pred)
        return score
    except ValueError:
        # Fallback for edge cases
        return 0.5
