import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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
    Computes the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    # Handle non-string inputs gracefully by converting to string
    str1 = str(str1) if str1 is not None else ""
    str2 = str(str2) if str2 is not None else ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)

    # Avoid division by zero if both strings are empty or contain no words
    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len


def compute_average_jaccard(ground_truths, predictions):
    """
    Computes the average Jaccard score for a list of ground truths and predictions.

    Args:
        ground_truths (list of str): List of ground truth answer strings.
        predictions (list of str): List of predicted answer strings.

    Returns:
        float: The average Jaccard score.
    """
    if not ground_truths or not predictions:
        return 0.0

    if len(ground_truths) != len(predictions):
        raise ValueError("Ground truths and predictions must have the same length.")

    scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)
