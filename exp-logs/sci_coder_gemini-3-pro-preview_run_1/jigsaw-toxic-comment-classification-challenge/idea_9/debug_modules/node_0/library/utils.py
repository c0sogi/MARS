import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def get_score(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.array): Ground truth labels of shape (N, num_labels).
        y_pred (np.array): Predicted probabilities of shape (N, num_labels).

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    scores = []
    # Inferred number of labels from the input shape
    num_labels = y_true.shape[1]

    for i in range(num_labels):
        try:
            # Calculate ROC AUC for the current column
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except ValueError:
            # This can happen if a batch/subset has only one class for a specific label.
            # In a full validation set with proper stratification, this should not occur.
            pass

    if len(scores) == 0:
        return 0.0

    return np.mean(scores)
