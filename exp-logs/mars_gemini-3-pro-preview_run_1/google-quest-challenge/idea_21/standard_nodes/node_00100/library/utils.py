import os
import random
import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

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
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape (N, num_targets).
        targets (np.ndarray or torch.Tensor): Ground truth targets of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Check shapes
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        # spearmanr returns a SpearmanRResult object or tuple (correlation, pvalue)
        # We access the correlation statistic.
        # Note: if the input is constant, correlation is undefined (NaN).
        # We'll treat NaN as 0.0 to avoid crashing.
        try:
            res = stats.spearmanr(preds[:, i], targets[:, i])
            # Handle different scipy versions or return types
            corr = res.correlation if hasattr(res, "correlation") else res[0]
        except Exception:
            corr = 0.0

        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
