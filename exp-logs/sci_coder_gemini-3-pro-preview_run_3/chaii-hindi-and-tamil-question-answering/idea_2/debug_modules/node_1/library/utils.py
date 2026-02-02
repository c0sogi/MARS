import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.
    Implementation based on the provided task metric description.

    Args:
        str1 (str): Ground truth string.
        str2 (str): Predicted string.

    Returns:
        float: Jaccard score.
    """
    # Ensure inputs are strings to prevent attribute errors
    str1 = str(str1) if str1 is not None else ""
    str2 = str(str2) if str2 is not None else ""

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
    Computes the average Jaccard score for a list of ground truths and predictions.

    Args:
        y_true (list): List of ground truth strings.
        y_pred (list): List of predicted strings.

    Returns:
        float: Average Jaccard score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: Ground truth has {len(y_true)} items, Predictions have {len(y_pred)} items."
        )

    if len(y_true) == 0:
        return 0.0

    scores = [jaccard(gt, pred) for gt, pred in zip(y_true, y_pred)]
    return sum(scores) / len(scores)


def clean_text(text):
    """
    Basic text cleaning helper to strip whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    return text.strip()
