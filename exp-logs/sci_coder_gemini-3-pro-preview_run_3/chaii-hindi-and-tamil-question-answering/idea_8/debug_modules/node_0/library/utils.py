import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denominator = len(a) + len(b) - len(c)

    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator
