import os
import random
import numpy as np
import torch
import nltk
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Useful for tracking loss and accuracy during training.
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


def calc_levenshtein(predictions, targets):
    """
    Calculates the mean Levenshtein distance between a list of prediction strings
    and a list of target strings.

    Args:
        predictions (list of str): List of predicted InChI strings.
        targets (list of str): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    distances = []
    for pred, target in zip(predictions, targets):
        # Calculate edit distance for each pair
        d = nltk.edit_distance(pred, target)
        distances.append(d)

    return np.mean(distances)


def save_checkpoint(state, is_best, filename=Config.MODEL_SAVE_PATH):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Path to save the best model. Defaults to Config.MODEL_SAVE_PATH.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the 'last' checkpoint for potential resumption
    last_path = os.path.join(directory, "last_model.pth")
    torch.save(state, last_path)

    # If this is the best model, save it to the specific best model path
    if is_best:
        torch.save(state, filename)
