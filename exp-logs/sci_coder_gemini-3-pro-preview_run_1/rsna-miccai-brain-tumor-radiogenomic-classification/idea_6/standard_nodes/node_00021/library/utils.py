import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Array-like of ground truth labels (0 or 1). Can be List, Numpy Array, or Tensor.
        y_pred: Array-like of predicted probabilities. Can be List, Numpy Array, or Tensor.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Convert PyTorch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge case where only one class is present in y_true
    # roc_auc_score raises a ValueError if only one class is present
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        score = roc_auc_score(y_true, y_pred)
        return score
    except ValueError:
        return 0.5
