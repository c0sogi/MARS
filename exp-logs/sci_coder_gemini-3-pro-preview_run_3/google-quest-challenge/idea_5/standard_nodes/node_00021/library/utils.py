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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        y_true (np.array or torch.Tensor): Ground truth target values. Shape (N, num_targets).
        y_pred (np.array or torch.Tensor): Predicted probabilities. Shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle 1D inputs by reshaping
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_cols = y_true.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        # Get the column vectors
        true_col = y_true[:, col_idx]
        pred_col = y_pred[:, col_idx]

        # Compute Spearman correlation
        # spearmanr returns a Result object (correlation, pvalue)
        try:
            corr, _ = spearmanr(true_col, pred_col)
        except Exception:
            corr = np.nan

        # Handle NaN values (e.g., if a column is constant, standard deviation is 0)
        # In such cases, correlation is undefined. We treat it as 0.0 for the mean.
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return float(np.mean(correlations))
