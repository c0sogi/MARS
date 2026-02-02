import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def compute_score(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true (array-like): Ground truth scores.
        y_pred (array-like): Predicted scores.

    Returns:
        float: The Pearson correlation coefficient. Returns 0.0 if calculation is undefined
               (e.g., constant input).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}."
        )

    if len(y_true) < 2:
        return 0.0

    # Check for zero variance to avoid division by zero or NaN results
    if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return 0.0

    return np.corrcoef(y_true, y_pred)[0, 1]


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
