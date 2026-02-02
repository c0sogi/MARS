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
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like or torch.Tensor): True binary labels (0 or 1).
        y_pred (array-like or torch.Tensor): Predicted probabilities for class 1.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC
    # We use a try-except block to handle cases where y_true might only have one class
    # which causes roc_auc_score to raise a ValueError.
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # If only one class is present, AUC is undefined. Return 0.5 as a neutral score.
        return 0.5
