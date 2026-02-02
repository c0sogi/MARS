import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure fully reproducible results.

    Args:
        seed (int): The random seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(preds, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_targets).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_cols = preds.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        # Extract columns
        pred_col = preds[:, col_idx]
        target_col = targets[:, col_idx]

        # Calculate Spearman's correlation
        # scipy.stats.spearmanr returns a SignificanceResult object or tuple
        res = spearmanr(target_col, pred_col)

        # Handle different scipy versions (object vs tuple)
        if hasattr(res, "statistic"):
            corr = res.statistic
        else:
            corr = res[0]

        # Handle NaN values (can occur if a column is constant)
        # We treat undefined correlation as 0.0 for metric stability
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    # Return the mean of column-wise correlations
    return np.mean(correlations)
