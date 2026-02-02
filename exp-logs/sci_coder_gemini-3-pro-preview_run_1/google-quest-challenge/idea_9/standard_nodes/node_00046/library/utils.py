import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def compute_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth target values of shape (N, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted target values of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
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

    num_cols = y_true.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        # Extract columns
        true_col = y_true[:, col_idx]
        pred_col = y_pred[:, col_idx]

        # Compute Spearman's correlation
        # spearmanr returns a tuple (correlation, pvalue) or an object where the first element is correlation
        # We handle the case where the result might be NaN (e.g., constant input) by replacing with 0 or ignoring,
        # but standard competition metrics usually expect valid inputs.
        # If correlation is nan, we treat it as 0.0 for summation purposes to avoid propagating NaN.
        corr = spearmanr(true_col, pred_col)[0]

        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
