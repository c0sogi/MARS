import numpy as np
import torch
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    Args:
        actual: A list of elements that are to be predicted (ground truth).
                (order doesn't matter in the lists)
        predicted: A list of predicted elements (order matters in the lists).
        k: The maximum number of predicted elements.

    Returns:
        The average precision at k over the input lists.
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
    Computes the mean average precision at k.

    Args:
        actual: A list of lists of elements that are to be predicted.
        predicted: A list of lists of predicted elements.
        k: The maximum number of predicted elements.

    Returns:
        The mean average precision at k over the input lists.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
