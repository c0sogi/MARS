import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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
    Used for tracking metrics like loss during training.
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
    Computes the Average Precision at k (AP@k) for a single sample.

    This implementation is optimized for single-label classification (one ground truth per image).

    Args:
        actual: The ground truth label (scalar).
        predicted: A list or array of predicted labels (ordered by confidence).
        k (int): The maximum number of predicted elements.

    Returns:
        float: The AP@k score.
    """
    # Truncate predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    # For single-label classification, the score is 1/rank if the actual label
    # is found within the top k predictions, otherwise 0.
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1.0)

    return 0.0


def mapk(actual, predicted, k=5):
    """
    Computes the Mean Average Precision at k (MAP@k).

    Args:
        actual: A list or array of ground truth labels.
        predicted: A list or array of lists of predicted labels.
        k (int): The maximum number of predicted elements.

    Returns:
        float: The MAP@k score.
    """
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same.")

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
