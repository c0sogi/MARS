import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed):
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


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array or list): Ground truth binary labels.
        y_pred (np.array or list): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays for compatibility
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if we have both classes to avoid sklearn errors
    if len(np.unique(y_true)) < 2:
        # If only one class is present, AUC is undefined/not useful.
        # Returning 0.5 as a neutral score or 0.0 depending on preference.
        # Here we return 0.5 to indicate random performance.
        return 0.5

    return roc_auc_score(y_true, y_pred)
