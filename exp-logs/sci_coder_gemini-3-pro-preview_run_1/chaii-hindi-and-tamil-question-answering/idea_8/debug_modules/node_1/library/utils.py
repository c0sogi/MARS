import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across runs.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1: str, str2: str) -> float:
    """
    Calculates the word-level Jaccard similarity score between two strings.

    The formula is: |A intersection B| / |A union B|
    where A and B are sets of words from str1 and str2 respectively.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)

    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len
