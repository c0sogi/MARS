import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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
        y_true (np.ndarray or pd.DataFrame): Ground truth target values.
        y_pred (np.ndarray or pd.DataFrame): Predicted target values.

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Convert pandas objects to numpy arrays if necessary
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_cols = y_true.shape[1]
    corrs = []

    for i in range(n_cols):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # Compute Spearman correlation
        # spearmanr returns a result object (with .statistic) or a tuple.
        # Index 0 is consistently the correlation coefficient.
        try:
            res = spearmanr(col_true, col_pred)
            if hasattr(res, "statistic"):
                corr = res.statistic
            else:
                corr = res[0]
        except Exception:
            # Handle edge cases, e.g., if inputs are constant
            corr = np.nan

        corrs.append(corr)

    # Calculate mean, ignoring NaNs if any (e.g. from constant columns)
    score = np.nanmean(corrs)
    return float(score)
