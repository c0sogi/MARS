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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard similarity between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    # Ensure inputs are strings to avoid attribute errors on split
    str1 = str(str1) if str1 is not None else ""
    str2 = str(str2) if str2 is not None else ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def compute_average_jaccard(ground_truths, predictions):
    """
    Computes the average Jaccard score for a list of ground truths and predictions.

    Args:
        ground_truths (list of str): List of ground truth answer strings.
        predictions (list of str): List of predicted answer strings.

    Returns:
        float: The average Jaccard score.
    """
    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"Length mismatch: ground_truths ({len(ground_truths)}) vs predictions ({len(predictions)})"
        )

    if len(ground_truths) == 0:
        return 0.0

    scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)


def save_score(score, path):
    """
    Saves a score to a file for regression gating.

    Args:
        score (float): The score to save.
        path (str): The file path.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(str(score))
    except Exception as e:
        print(f"Warning: Could not save score to {path}. Error: {e}")


def load_score(path):
    """
    Loads a score from a file. Returns 0.0 if file doesn't exist or is invalid.

    Args:
        path (str): The file path.

    Returns:
        float: The loaded score.
    """
    if not os.path.exists(path):
        return 0.0

    try:
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return 0.0
            return float(content)
    except Exception:
        return 0.0
