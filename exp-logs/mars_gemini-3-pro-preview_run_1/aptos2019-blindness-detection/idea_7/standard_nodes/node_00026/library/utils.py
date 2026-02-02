import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for CuDNN
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


def compute_score(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels or continuous scores.

    Returns:
        float: The quadratic weighted kappa score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If predictions are floating point (e.g., from regression/ordinal sum),
    # round to the nearest integer as QWK expects discrete ratings.
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(y_pred).astype(int)

    # Clip predictions to the valid range [0, 4] to ensure metric validity
    y_pred = np.clip(y_pred, 0, 4)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filename, best_filename)
