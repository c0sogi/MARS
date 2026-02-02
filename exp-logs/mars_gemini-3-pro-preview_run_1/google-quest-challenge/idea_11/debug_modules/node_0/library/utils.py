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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
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


def compute_spearman_correlation(preds, targets):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        preds (np.ndarray): Predicted probabilities, shape (N, num_targets).
        targets (np.ndarray): Ground truth labels, shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    num_targets = preds.shape[1]
    correlations = []

    for i in range(num_targets):
        pred_col = preds[:, i]
        target_col = targets[:, i]

        # Check for constant values to avoid NaNs/warnings from spearmanr
        # If a column is constant, correlation is technically undefined, usually treated as 0 in this context
        if np.std(pred_col) == 0 or np.std(target_col) == 0:
            corr = 0.0
        else:
            # spearmanr returns a Result object (statistic, pvalue) or tuple
            # We take the statistic (index 0)
            corr = spearmanr(pred_col, target_col)[0]

            # Handle NaN result if it slips through
            if np.isnan(corr):
                corr = 0.0

        correlations.append(corr)

    return np.mean(correlations)
