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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).
    Handles PyTorch tensors by converting them to NumPy arrays.

    Args:
        y_true (array-like or torch.Tensor): Ground truth binary labels.
        y_pred (array-like or torch.Tensor): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
