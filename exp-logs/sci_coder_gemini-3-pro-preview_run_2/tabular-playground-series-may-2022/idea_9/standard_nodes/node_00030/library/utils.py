import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Handles both NumPy arrays and PyTorch tensors as input.

    Args:
        y_true (array-like or torch.Tensor): Ground truth binary labels.
        y_pred (array-like or torch.Tensor): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5
