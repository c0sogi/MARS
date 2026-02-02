import os
import random
import numpy as np
import torch


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    The formula is: intersection(words) / union(words).

    Args:
        str1 (str): The first string (typically ground truth).
        str2 (str): The second string (typically prediction).

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    # Ensure inputs are strings to avoid AttributeError on .lower()
    str1 = str(str1) if str1 is not None else ""
    str2 = str(str2) if str2 is not None else ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)

    # Avoid division by zero if both sets are empty
    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
