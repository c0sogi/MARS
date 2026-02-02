import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def compute_score(y_true, y_pred):
    """
    Computes the average Jaccard score for a list of predictions.

    Args:
        y_true (list of str): List of ground truth strings.
        y_pred (list of str): List of predicted strings.

    Returns:
        float: The average Jaccard score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of ground truth and predictions must match.")

    if len(y_true) == 0:
        return 0.0

    scores = [jaccard(gt, pred) for gt, pred in zip(y_true, y_pred)]
    return sum(scores) / len(scores)
