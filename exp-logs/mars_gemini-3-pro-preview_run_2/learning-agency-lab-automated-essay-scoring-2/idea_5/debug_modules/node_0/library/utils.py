import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    This function wraps sklearn.metrics.cohen_kappa_score with weights='quadratic'.
    It handles PyTorch tensors by detaching and converting them to NumPy arrays.
    It also ensures inputs are integers, as required by the kappa metric.

    Args:
        y_true: Array-like or Tensor of true values (expected to be integers 1-6).
        y_pred: Array-like or Tensor of predicted values.

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Handle PyTorch Tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # The kappa metric requires discrete integer labels.
    # If floats are passed (e.g., raw regression outputs), we round them.
    # Note: Ideally, threshold optimization should be applied before calling this
    # if the model outputs continuous scores.
    if y_true.dtype.kind == "f":
        y_true = np.round(y_true).astype(int)
    if y_pred.dtype.kind == "f":
        y_pred = np.round(y_pred).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
