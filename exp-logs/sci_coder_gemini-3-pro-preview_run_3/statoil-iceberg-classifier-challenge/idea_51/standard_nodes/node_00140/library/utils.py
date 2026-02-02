import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current fold number (used for naming).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.WORKING_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def log_loss_score(y_true, y_pred):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # sklearn log_loss handles clipping internally (eps=1e-15 by default)
    # to avoid log(0) errors.
    return log_loss(y_true, y_pred)
