import os
import math
import time
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
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


def get_score(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: The mean ROC AUC score.
    """
    scores = []
    num_columns = y_true.shape[1]

    for i in range(num_columns):
        # Check if column has both classes (0 and 1)
        # roc_auc_score throws ValueError if only one class is present
        if len(np.unique(y_true[:, i])) == 2:
            try:
                col_score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(col_score)
            except ValueError:
                pass

    if len(scores) > 0:
        return np.mean(scores)
    else:
        return 0.5


def as_minutes(s):
    """
    Converts seconds to a string format 'm m s s'.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    """
    Calculates estimated time remaining based on progress.

    Args:
        since (float): Time when the process started.
        percent (float): Current progress percentage (0.0 to 1.0).

    Returns:
        str: Formatted string showing elapsed time and estimated remaining time.
    """
    now = time.time()
    s = now - since
    if percent > 0:
        es = s / percent
        rs = es - s
    else:
        rs = 0
    return "%s (remain %s)" % (as_minutes(s), as_minutes(rs))
