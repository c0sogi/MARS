import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="model_best.pth"
):
    """
    Saves the current model state to a checkpoint file.
    If the current model is the best one, it copies the file to best_filename.

    Args:
        state (dict): The model state dictionary (and optimizer state, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Path to save the current checkpoint.
        best_filename (str): Path to save the best model.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)
    if is_best:
        # Ensure the directory for best_filename exists as well (if different)
        best_directory = os.path.dirname(best_filename)
        if best_directory:
            os.makedirs(best_directory, exist_ok=True)
        shutil.copyfile(filename, best_filename)


def calculate_metrics(y_true, y_pred):
    """
    Calculates the Macro F1 score for the given predictions.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")
