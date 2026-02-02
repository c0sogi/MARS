import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from library.config import seed_everything


def compute_spearman_metric(y_true, y_pred, target_cols=None):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true: Ground truth targets. Can be numpy array, pandas DataFrame, or torch Tensor.
        y_pred: Predicted targets. Can be numpy array, pandas DataFrame, or torch Tensor.
        target_cols: Optional list of column names for mapping scores (unused in calculation but kept for API compatibility).

    Returns:
        float: The mean column-wise Spearman's correlation.
    """
    # Convert torch tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert pandas to numpy
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Compute Spearman correlation for each column
    correlations = []
    num_cols = y_true.shape[1]

    for i in range(num_cols):
        # Extract columns
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        # Check for constant columns to avoid warnings/NaNs
        # Spearman correlation is undefined if one variable is constant
        if np.std(true_col) == 0 or np.std(pred_col) == 0:
            corr = 0.0
        else:
            corr, _ = spearmanr(true_col, pred_col)
            # Handle NaN if it still occurs (e.g., numerical instability)
            if np.isnan(corr):
                corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
