import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve.
    Handles cases where specific classes may not be present in the ground truth
    by calculating AUC per class and averaging only over valid classes.

    Args:
        y_true (np.ndarray): Ground truth labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Only calculate AUC if the class has both positive and negative samples
        # This prevents ValueError when a class is completely absent in the validation batch/set
        if len(np.unique(y_true[:, i])) == 2:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(auc)
            except ValueError:
                pass

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)
