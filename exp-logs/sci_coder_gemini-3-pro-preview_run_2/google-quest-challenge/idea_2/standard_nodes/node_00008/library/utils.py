import os
import random
import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(predictions, targets):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        predictions: Numpy array or PyTorch tensor of shape (N, num_targets).
                     Values should be continuous probabilities or scores.
        targets: Numpy array or PyTorch tensor of shape (N, num_targets).
                 Ground truth values.

    Returns:
        float: The mean Spearman correlation across all target columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Check shapes
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    # Handle 1D arrays (single target) by reshaping
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    num_cols = predictions.shape[1]
    correlations = []

    for col_idx in range(num_cols):
        pred_col = predictions[:, col_idx]
        target_col = targets[:, col_idx]

        # Spearmanr returns (correlation, p-value). We only need the correlation.
        # We wrap in try-except to handle potential edge cases (e.g., constant input)
        try:
            corr, _ = stats.spearmanr(pred_col, target_col)
            # If the result is NaN (e.g. constant column), treat as 0 correlation
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
