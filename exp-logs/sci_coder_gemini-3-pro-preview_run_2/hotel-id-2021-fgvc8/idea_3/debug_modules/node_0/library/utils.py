import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
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
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to record.
            n (int): The weight/count of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k for a single sample.

    Args:
        actual (list): A list of elements that are to be predicted (ground truth).
        predicted (list): A list of predicted elements.
        k (int): The maximum number of predicted elements.

    Returns:
        float: The average precision at k.
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


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k for a list of samples.

    Args:
        actual (list of lists): A list of lists, where each inner list contains
                                the ground truth elements for a sample.
        predicted (list of lists): A list of lists of predicted elements.
        k (int): The maximum number of predicted elements.

    Returns:
        float: The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
