import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value. Defaults to the value in Config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true (array-like): The ground truth labels (integers).
        y_pred (array-like): The predicted labels (integers).

    Returns:
        float: The quadratic weighted kappa score.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class AverageMeter:
    """
    Computes and stores the average and current value.
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
