import numpy as np
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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
        actual: The ground truth value (can be a list or a single value).
        predicted: A list of predicted elements.
        k: The maximum number of predicted elements.

    Returns:
        The average precision at k.
    """
    # Ensure actual is a list
    if not isinstance(actual, list):
        actual = [actual]

    # Truncate predicted list to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # Check if prediction is in ground truth and not a duplicate prediction
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.

    Args:
        actual: A list of ground truth values (one per sample).
        predicted: A list of lists of predicted elements (one list per sample).
        k: The maximum number of predicted elements.

    Returns:
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
