import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N_samples, N_labels).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N_samples, N_labels).

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Check for shape mismatch
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_cols = preds.shape[1]
    corrs = []

    for col_idx in range(num_cols):
        p_col = preds[:, col_idx]
        t_col = targets[:, col_idx]

        # spearmanr returns a result object or tuple. Index 0 is the correlation.
        # We use a try-except block or check for nan to handle constant inputs gracefully.
        try:
            res = spearmanr(p_col, t_col)
            # Access correlation safely (handles both tuple and object returns)
            corr = res.correlation if hasattr(res, "correlation") else res[0]
        except Exception:
            corr = 0.0

        # If the input is constant, spearmanr returns NaN
        if np.isnan(corr):
            corr = 0.0

        corrs.append(corr)

    return np.mean(corrs)
