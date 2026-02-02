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


def normalize_text(text: str) -> str:
    """
    Normalizes text by collapsing multiple spaces into a single space
    and stripping leading/trailing whitespace. This enforces the
    'Normalize-First' protocol required for alignment.

    Args:
        text (str): The input text string.

    Returns:
        str: The normalized text string.
    """
    if not isinstance(text, str):
        return str(text)

    # Split by whitespace (handles multiple spaces, tabs, newlines)
    # and rejoin with a single space.
    return " ".join(text.split())


def jaccard(str1: str, str2: str) -> float:
    """
    Calculates the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): First string (e.g., predicted text).
        str2 (str): Second string (e.g., ground truth text).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)
    return float(len(c)) / union_len if union_len > 0 else 0.0


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and scores during training.
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
