import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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


def get_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays for safety, though sklearn handles lists
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle cases where only one class is present in y_true
        return 0.5
