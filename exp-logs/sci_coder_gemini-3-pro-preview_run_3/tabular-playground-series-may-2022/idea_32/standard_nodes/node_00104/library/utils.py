import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_roc_auc(y_true, y_score):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Handles inputs that may be PyTorch tensors by converting them to NumPy arrays.

    Args:
        y_true (array-like or torch.Tensor): True binary labels.
        y_score (array-like or torch.Tensor): Target scores or probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    return roc_auc_score(y_true, y_score)
