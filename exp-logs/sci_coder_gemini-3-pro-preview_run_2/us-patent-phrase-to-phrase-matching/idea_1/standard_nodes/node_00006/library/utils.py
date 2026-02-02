import numpy as np
from scipy.stats import pearsonr
from library.config import set_seed


def compute_pearson_correlation(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient between true and predicted scores.

    This function handles inputs that may be Python lists, NumPy arrays, or
    PyTorch tensors. It ensures they are flattened and on the CPU before
    calculation.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Handle PyTorch Tensors: detach and move to CPU if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert to numpy arrays and flatten to 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # scipy.stats.pearsonr returns (statistic, p-value)
    # We only need the statistic for the competition metric
    correlation, _ = pearsonr(y_true, y_pred)

    return correlation
