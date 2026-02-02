import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth target values of shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted target values of shape (N, C).

    Returns:
        float: The mean column-wise Spearman's correlation coefficient.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure they are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_cols = y_true.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        # spearmanr returns (correlation, p-value) or an object with statistic
        try:
            res = spearmanr(y_true[:, col_idx], y_pred[:, col_idx])
            # Handle both object return (newer scipy) and tuple return (older scipy)
            try:
                corr = res.statistic
            except AttributeError:
                corr = res[0]
        except Exception:
            corr = np.nan

        correlations.append(corr)

    # Filter out NaNs before averaging (NaNs can occur if a column is constant)
    valid_corrs = [c for c in correlations if not np.isnan(c)]

    if not valid_corrs:
        return 0.0

    return float(np.mean(valid_corrs))
