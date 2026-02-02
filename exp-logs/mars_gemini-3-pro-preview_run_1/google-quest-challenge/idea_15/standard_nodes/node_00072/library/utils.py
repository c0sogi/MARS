import os
import random
import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(preds, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        preds: Predictions array (N_samples, N_targets). Can be numpy array or torch tensor.
        targets: Ground truth array (N_samples, N_targets). Can be numpy array or torch tensor.

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are at least 2D
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    num_cols = preds.shape[1]
    corrs = []

    for col_idx in range(num_cols):
        p = preds[:, col_idx]
        t = targets[:, col_idx]

        # Calculate Spearman correlation
        # scipy.stats.spearmanr returns a Result object or tuple (correlation, pvalue)
        try:
            res = stats.spearmanr(p, t)
            if hasattr(res, "statistic"):
                # Scipy >= 1.7
                corr = res.statistic
            else:
                # Older Scipy
                corr = res[0]
        except Exception:
            corr = 0.0

        # Handle cases where correlation is NaN (e.g., constant input)
        if np.isnan(corr):
            corr = 0.0

        corrs.append(corr)

    return np.mean(corrs)
