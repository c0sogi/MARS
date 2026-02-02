import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def compute_spearman_metric(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_targets).

    Returns:
        float: The mean Spearman's rank correlation coefficient across all target columns.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    correlations = []
    num_targets = y_true.shape[1]

    for i in range(num_targets):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # spearmanr returns (correlation, pvalue) or an object that behaves like a tuple
        # Index 0 is the correlation coefficient
        try:
            res = spearmanr(col_true, col_pred)
            corr = res[0]
        except Exception:
            corr = np.nan

        correlations.append(corr)

    # Filter out NaNs (e.g., from constant columns)
    valid_correlations = [c for c in correlations if not np.isnan(c)]

    if not valid_correlations:
        return 0.0

    return np.mean(valid_correlations)
