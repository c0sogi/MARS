import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed value. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_spearman_correlation(preds, target):
    """
    Computes the mean column-wise Spearman's rank correlation coefficient.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_targets).
        target (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all columns.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure shapes match
    if preds.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs target {target.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        col_pred = preds[:, i]
        col_target = target[:, i]

        # Handle constant columns (zero variance) which cause division by zero in correlation calculation
        # If the standard deviation is effectively zero, correlation is undefined (NaN).
        # We treat this as 0.0 correlation.
        if np.std(col_pred) < 1e-9 or np.std(col_target) < 1e-9:
            corr = 0.0
        else:
            # spearmanr returns (correlation, pvalue) or an object where [0] is correlation
            try:
                res = spearmanr(col_pred, col_target)
                # Access correlation safely across different scipy versions
                corr = res[0] if isinstance(res, (tuple, list)) else res.correlation
            except Exception:
                corr = 0.0

        # Handle NaN values explicitly
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
