import os
import random
import shutil
import numpy as np
import torch
from library.config import CFG


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Enforces deterministic cudnn algorithms.

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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training epochs.
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


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k for a single sample.

    Args:
        actual: The ground truth label (string or int).
        predicted: A list of predicted labels (strings or ints).
        k: The maximum number of predicted items to consider.

    Returns:
        float: The average precision at k.
    """
    # Truncate predicted list to k
    if len(predicted) > k:
        predicted = predicted[:k]

    # For single-label classification, we check the rank of the correct label.
    # If the correct label is at index i (0-indexed), the precision is 1/(i+1).
    # If the correct label is not in the top k, the precision is 0.
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1.0)

    return 0.0


def calc_map5(actual_list, predicted_list):
    """
    Computes the Mean Average Precision @ 5 (MAP@5) across the dataset.

    Args:
        actual_list: List of ground truth labels.
        predicted_list: List of lists, where each inner list contains the top-k predictions.

    Returns:
        float: The MAP@5 score.
    """
    if not actual_list or not predicted_list:
        return 0.0

    scores = [apk(a, p, k=5) for a, p in zip(actual_list, predicted_list)]
    return np.mean(scores)


def save_checkpoint(state, is_best, filepath=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best metric so far.
        filepath (str, optional): Path to save the checkpoint. Defaults to CFG.model_path.
    """
    if filepath is None:
        filepath = CFG.model_path

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        # Save a copy as the best model
        # Assuming filepath ends with .pth, we insert _best before the extension
        base, ext = os.path.splitext(filepath)
        best_filepath = f"{base}_best{ext}"
        shutil.copyfile(filepath, best_filepath)
