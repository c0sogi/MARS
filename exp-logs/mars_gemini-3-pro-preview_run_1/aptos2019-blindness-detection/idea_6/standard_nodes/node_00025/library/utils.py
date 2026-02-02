import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    This metric measures the agreement between two ratings. This implementation
    handles PyTorch tensors and NumPy arrays. It assumes that y_pred might be
    continuous (e.g., from an ordinal regression sum) and rounds it to the
    nearest integer before calculating the score.

    Args:
        y_true: Ground truth labels (1D array-like).
        y_pred: Predicted labels or scores (1D array-like).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Round predictions to nearest integer for QWK calculation
    # This aligns with the strategy: Sum of probs -> continuous score -> round -> int class
    y_pred_rounded = np.rint(y_pred).astype(int)
    y_true_int = y_true.astype(int)

    # Calculate QWK using sklearn
    qwk = cohen_kappa_score(y_true_int, y_pred_rounded, weights="quadratic")

    return qwk
