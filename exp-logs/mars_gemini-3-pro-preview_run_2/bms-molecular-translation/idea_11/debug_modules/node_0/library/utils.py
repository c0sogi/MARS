import os
import random
import shutil
import numpy as np
import torch
from nltk.metrics.distance import edit_distance
from library.config import Config


class AverageMeter:
    """Computes and stores the average and current value."""

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


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (list of str): List of predicted InChI strings.
        targets (list of str): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    score = 0
    n = len(predictions)

    if n == 0:
        return 0.0

    for pred, target in zip(predictions, targets):
        score += edit_distance(pred, target)

    return score / n


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory defined in Config.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_path)
