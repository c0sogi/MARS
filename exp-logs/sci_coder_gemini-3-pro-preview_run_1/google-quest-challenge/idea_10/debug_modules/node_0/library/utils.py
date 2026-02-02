import os
import random
import numpy as np
import torch
from scipy import stats
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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


def compute_spearman_correlation(predictions, targets):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        predictions: (N, 30) numpy array or torch tensor of predicted probabilities.
        targets: (N, 30) numpy array or torch tensor of ground truth labels.

    Returns:
        float: The mean Spearman correlation across all target columns.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    num_targets = predictions.shape[1]
    corrs = []

    for i in range(num_targets):
        # Extract columns
        pred_col = predictions[:, i]
        target_col = targets[:, i]

        # Calculate Spearman correlation
        # scipy.stats.spearmanr returns an object with a 'statistic' attribute in newer versions,
        # or a tuple (correlation, pvalue) in older versions.
        res = stats.spearmanr(pred_col, target_col)

        try:
            corr = res.statistic
        except AttributeError:
            # Fallback for older scipy versions or if result is a tuple
            if isinstance(res, (tuple, list)):
                corr = res[0]
            else:
                corr = res

        corrs.append(corr)

    # Return mean, ignoring NaNs (which can occur if a column is constant)
    return np.nanmean(corrs)
