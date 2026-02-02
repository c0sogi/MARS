import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
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


def compute_pearson_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted values.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The Pearson correlation coefficient.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # np.corrcoef returns the correlation matrix
    # [[Var(x), Cov(x,y)], [Cov(y,x), Var(y)]] normalized
    # We want the off-diagonal element [0, 1]
    return np.corrcoef(y_true, y_pred)[0, 1]
