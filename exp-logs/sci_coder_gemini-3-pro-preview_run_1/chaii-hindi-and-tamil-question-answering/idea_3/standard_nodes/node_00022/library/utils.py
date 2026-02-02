import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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
    Calculates the word-level Jaccard similarity score between two strings.

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

    # Handle edge case where both sets are empty to avoid division by zero
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def compute_score(y_true, y_pred):
    """
    Computes the average Jaccard score for a list of ground truth and predicted strings.

    Args:
        y_true (list of str): List of ground truth answer strings.
        y_pred (list of str): List of predicted answer strings.

    Returns:
        float: The average Jaccard score across all samples.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch between ground truth ({len(y_true)}) and predictions ({len(y_pred)})."
        )

    scores = [jaccard(gt, pred) for gt, pred in zip(y_true, y_pred)]
    return np.mean(scores)
