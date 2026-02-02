import os
import shutil
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CuDNN backends.
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
    Useful for tracking losses and metrics over batches.
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
    Computes the average precision at k.

    Parameters
    ----------
    actual : list
        A list of elements that are to be predicted (ground truth).
        For this task, typically a list containing a single label.
    predicted : list
        A list of predicted elements (order matters).
    k : int, optional
        The maximum number of predicted elements.

    Returns
    -------
    score : double
        The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    actual : list
        A list of lists of elements that are to be predicted.
    predicted : list
        A list of lists of predicted elements.
    k : int, optional
        The maximum number of predicted elements.

    Returns
    -------
    score : double
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map5(actual, predicted):
    """
    Wrapper for mapk with k=5, specifically for the competition metric.

    Parameters
    ----------
    actual : list
        A list of lists, where each inner list contains the ground truth label(s).
    predicted : list
        A list of lists, where each inner list contains the top 5 predicted labels.
    """
    return mapk(actual, predicted, k=5)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", result_dir=None):
    """
    Saves the model checkpoint.

    Parameters
    ----------
    state : dict
        State dictionary containing model parameters, optimizer state, etc.
    is_best : bool
        Whether this checkpoint represents the best model so far.
    filename : str
        Name of the checkpoint file.
    result_dir : str, optional
        Directory to save the checkpoint. Defaults to Config.WORKING_DIR.
    """
    if result_dir is None:
        result_dir = Config.WORKING_DIR

    os.makedirs(result_dir, exist_ok=True)

    filepath = os.path.join(result_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(result_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)
