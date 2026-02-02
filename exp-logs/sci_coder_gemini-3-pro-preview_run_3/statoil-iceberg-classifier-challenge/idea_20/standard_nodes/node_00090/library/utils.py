import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric (Binary Cross Entropy).

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # Scikit-learn's log_loss handles epsilon clipping internally to avoid log(0)
    return log_loss(y_true, y_pred)


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the configured directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current cross-validation fold index.
    """
    # Ensure the directory exists (redundant if Config handles it, but safe)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filename = os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )
        torch.save(state, best_filename)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training iterations.
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
