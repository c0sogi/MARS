import os
import random
import numpy as np
import torch
from scipy import stats
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearman_metric(preds, targets):
    """
    Computes the Mean Column-wise Spearman's Rank Correlation Coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, 30).
        targets (np.ndarray or torch.Tensor): Ground truth labels of shape (N, 30).

    Returns:
        float: The mean Spearman's correlation across all target columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {preds.shape} vs targets {targets.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        pred_col = preds[:, i]
        target_col = targets[:, i]

        # Calculate Spearman's correlation
        # spearmanr returns an object with a 'statistic' attribute in newer scipy versions,
        # or a tuple (correlation, p-value) in older versions.
        res = stats.spearmanr(pred_col, target_col)

        try:
            corr = res.statistic
        except AttributeError:
            corr = res[0]

        # Handle cases where correlation is NaN (e.g., constant prediction or target)
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
