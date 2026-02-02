import os
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

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


def get_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted values.

    Args:
        y_true (np.array or list): Ground truth scores.
        y_pred (np.array or list): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    score = pearsonr(y_true, y_pred)[0]
    return score


class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The number of samples associated with this value (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
