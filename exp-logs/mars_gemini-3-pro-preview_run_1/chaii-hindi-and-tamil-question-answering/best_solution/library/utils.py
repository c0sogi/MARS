import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1: str, str2: str) -> float:
    """
    Computes the word-level Jaccard similarity score between two strings.

    Formula: Intersection over Union of the set of words.

    Args:
        str1 (str): The ground truth string.
        str2 (str): The predicted string.

    Returns:
        float: The Jaccard similarity score [0.0, 1.0].
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)

    if union_len == 0:
        # If both sets are empty, they are identical (score 1.0).
        if len(a) == 0 and len(b) == 0:
            return 1.0
        return 0.0

    return float(len(c)) / union_len


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
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
