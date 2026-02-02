import os
import random
import numpy as np
import torch
from scipy.stats import spearmanr


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


def compute_spearmanr(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray): Ground truth target values of shape (N, num_targets).
        y_pred (np.ndarray): Predicted target values of shape (N, num_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all target columns.
    """
    corrs = []
    # Iterate over each column (target variable)
    for col_idx in range(y_true.shape[1]):
        # Calculate Spearman's correlation for the current column
        # spearmanr returns an object with a 'statistic' attribute
        # We assume input arrays are flat for each column or flatten them implicitly
        true_col = y_true[:, col_idx]
        pred_col = y_pred[:, col_idx]

        # Handle constant input cases to avoid NaNs if necessary,
        # though scipy usually handles this or returns NaN.
        # Given the task description, we proceed with standard calculation.
        corr_result = spearmanr(true_col, pred_col)

        # Access the correlation statistic
        # Check if result is a tuple (old scipy) or object (new scipy)
        if hasattr(corr_result, "statistic"):
            corr = corr_result.statistic
        else:
            corr = corr_result[0]

        corrs.append(corr)

    return np.nanmean(corrs)
