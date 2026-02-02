import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Seeds all random number generators to ensure reproducibility.

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


def jaccard(str1, str2):
    """
    Computes the Jaccard score between two strings.

    Args:
        str1 (str): First string (e.g., predicted text).
        str2 (str): Second string (e.g., ground truth text).

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    # Edge case handling based on analysis script logic
    if (len(a) == 0) and (len(b) == 0):
        return 0.5

    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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
