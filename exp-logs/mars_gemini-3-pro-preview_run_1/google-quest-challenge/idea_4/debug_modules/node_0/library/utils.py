import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth target values. Shape (N, 30).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities. Shape (N, 30).

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    num_cols = y_true.shape[1]
    correlations = []

    for col in range(num_cols):
        # Extract columns
        t = y_true[:, col]
        p = y_pred[:, col]

        # Compute Spearman correlation
        # spearmanr returns a tuple (correlation, p-value) or an object behaving like one
        # We take the first element (correlation)
        try:
            corr = spearmanr(t, p)[0]
        except Exception:
            corr = 0.0

        # Handle NaNs (can happen if a column is constant)
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
