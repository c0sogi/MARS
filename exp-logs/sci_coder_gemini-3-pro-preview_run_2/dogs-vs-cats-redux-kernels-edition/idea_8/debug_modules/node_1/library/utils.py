import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
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
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1 (Dog).

    Returns:
        float: The calculated log loss.
    """
    # sklearn log_loss handles binary classification where y_pred is probability of positive class
    # It also handles clipping internally to avoid log(0)
    return log_loss(y_true, y_pred, labels=[0, 1])
