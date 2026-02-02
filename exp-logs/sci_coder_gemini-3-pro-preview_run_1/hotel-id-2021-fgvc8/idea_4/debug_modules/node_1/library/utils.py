import numpy as np
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
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
        actual: The ground truth value. Can be a scalar (single class ID) or a list of relevant items.
        predicted: A list of predicted elements (ordered by confidence).
        k: The maximum number of predicted elements to consider.

    Returns:
        The average precision at k.
    """
    # Normalize actual to a list if it is a scalar
    if not isinstance(actual, (list, np.ndarray, set)):
        actual = [actual]

    # Truncate predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # Check if prediction is relevant and not a duplicate in predictions
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    # AP@k is the sum of precisions divided by the number of relevant items (capped at k)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k across all samples.

    Args:
        actual: A list of ground truth values (scalars or lists).
        predicted: A list of lists of predicted elements.
        k: The maximum number of predicted elements to consider.

    Returns:
        The mean average precision at k.
    """
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same.")

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
