import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
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


def jaccard(str1, str2):
    """
    Calculates the Jaccard similarity score between two strings at the word level.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        float: The Jaccard similarity score (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    # If both sets are empty, define similarity as 0.5 (neutral/undefined overlap)
    # consistent with common evaluation scripts for this task
    if (len(a) == 0) & (len(b) == 0):
        return 0.5

    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
