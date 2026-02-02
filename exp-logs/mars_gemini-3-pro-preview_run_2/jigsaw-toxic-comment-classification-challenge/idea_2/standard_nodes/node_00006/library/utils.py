import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true: Array-like of shape (n_samples, n_classes) containing true binary labels.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # average='macro' calculates the metric for each label, and finds their unweighted mean.
    # This matches the competition metric "Mean column-wise ROC AUC".
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Fallback for cases where a batch might contain only one class for a specific label
        # In a full validation set this shouldn't happen due to stratification, but helpful for debugging
        return 0.5


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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
