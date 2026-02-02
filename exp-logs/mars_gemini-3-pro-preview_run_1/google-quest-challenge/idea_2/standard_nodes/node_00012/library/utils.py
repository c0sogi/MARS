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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(predictions, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        predictions: np.array or torch.Tensor of shape (N, num_labels).
                     Predicted probabilities or logits.
        targets: np.array or torch.Tensor of shape (N, num_labels).
                 Ground truth labels.

    Returns:
        float: The mean Spearman correlation across all columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure shapes match
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    num_cols = predictions.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        pred_col = predictions[:, col_idx]
        target_col = targets[:, col_idx]

        # Compute Spearman correlation
        # spearmanr returns an object with a .correlation attribute
        # If inputs are constant, it may return NaN
        try:
            corr = spearmanr(pred_col, target_col).correlation
        except Exception:
            corr = np.nan

        correlations.append(corr)

    # Calculate mean, ignoring NaNs (e.g., from constant columns in a batch)
    mean_corr = np.nanmean(correlations)

    return float(mean_corr)
