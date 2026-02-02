import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    c = a.intersection(b)

    if (len(a) + len(b) - len(c)) > 0:
        return float(len(c)) / (len(a) + len(b) - len(c))
    else:
        # If both sets are empty, they are identical (score 1.0).
        # If one is empty and the other is not, the denominator is > 0, handled above.
        # This case (0/0) implies both are empty.
        return 1.0


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
    Calculates the average Jaccard score for a batch of predictions.

    Args:
        y_true (list of str): List of ground truth strings.
        y_pred (list of str): List of predicted strings.

    Returns:
        float: The average Jaccard score.
    """
    scores = []
    for i in range(len(y_true)):
        score = jaccard(y_true[i], y_pred[i])
        scores.append(score)

    return np.mean(scores)
