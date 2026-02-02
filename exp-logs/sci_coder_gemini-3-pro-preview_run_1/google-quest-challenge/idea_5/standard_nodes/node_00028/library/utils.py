import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr_score(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.array or torch.Tensor): Predicted probabilities of shape (n_samples, n_targets).
        targets (np.array or torch.Tensor): Ground truth labels of shape (n_samples, n_targets).

    Returns:
        float: The mean Spearman's correlation coefficient.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Check shapes
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    score_list = []
    # Iterate over each target column
    for i in range(targets.shape[1]):
        # Calculate Spearman's correlation for the current column
        # spearmanr returns a Result object or tuple (correlation, p-value)
        # We extract the correlation coefficient
        col_preds = preds[:, i]
        col_targets = targets[:, i]

        # Handle constant columns to avoid warnings/errors if necessary,
        # though spearmanr typically returns NaN for constant inputs.
        corr, _ = spearmanr(col_preds, col_targets)
        score_list.append(corr)

    # Calculate the mean of the correlation coefficients
    # Use np.nanmean to handle potential NaNs (e.g., if a column was constant)
    score = np.nanmean(score_list)

    return score
