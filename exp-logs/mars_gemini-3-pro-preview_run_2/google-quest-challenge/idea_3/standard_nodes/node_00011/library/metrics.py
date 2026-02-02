import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def compute_spearmanr(y_true, y_pred):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    This metric is calculated by computing the Spearman's rank correlation coefficient
    for each target column independently and then taking the mean of these values.

    Args:
        y_true: Ground truth labels. Shape (N_samples, N_targets).
                Can be a numpy array or a torch.Tensor.
        y_pred: Predicted probabilities. Shape (N_samples, N_targets).
                Can be a numpy array or a torch.Tensor.

    Returns:
        float: The mean column-wise Spearman's correlation coefficient.
    """
    # Handle PyTorch Tensors: detach from graph and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Basic Shape Validation
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D arrays. Got dims: y_true={y_true.ndim}, y_pred={y_pred.ndim}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Spearman's correlation for each column
    correlations = []
    num_cols = y_true.shape[1]

    for col_idx in range(num_cols):
        col_true = y_true[:, col_idx]
        col_pred = y_pred[:, col_idx]

        # Check for constant columns (zero variance)
        # Spearman correlation is undefined if one variable is constant.
        # We assign a correlation of 0.0 in these cases to avoid NaNs.
        if np.std(col_true) == 0 or np.std(col_pred) == 0:
            corr = 0.0
        else:
            # spearmanr returns a result object or tuple depending on scipy version
            res = spearmanr(col_true, col_pred)
            try:
                corr = res.statistic
            except AttributeError:
                # Fallback for older scipy versions where result is a tuple (corr, pvalue)
                corr = res[0]

        # Explicitly handle NaN results if they slip through
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    # Return the mean of the column-wise correlations
    return float(np.mean(correlations))
