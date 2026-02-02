import os
import random
import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42):
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


def compute_spearmanr(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds: Numpy array or Torch tensor of predictions (N_samples, N_targets).
        targets: Numpy array or Torch tensor of ground truth labels (N_samples, N_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        pred_col = preds[:, i]
        target_col = targets[:, i]

        # scipy.stats.spearmanr returns a SignificanceResult object or tuple.
        # Index [0] is the correlation coefficient.
        # We compute correlation for each column pair individually.
        corr = stats.spearmanr(target_col, pred_col)[0]

        # Handle NaN values which can occur if a column is constant (std dev is 0)
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
