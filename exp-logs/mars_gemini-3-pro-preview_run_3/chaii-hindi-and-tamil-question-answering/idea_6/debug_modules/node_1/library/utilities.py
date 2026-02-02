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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    if (len(a) + len(b) - len(c)) == 0:
        return 0.0

    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_average_jaccard(ground_truths, predictions):
    """
    Computes the average Jaccard score for a list of ground truths and predictions.
    Formula: score = (1/n) * sum(jaccard(gt_i, dt_i))

    Args:
        ground_truths (list of str): List of ground truth strings.
        predictions (list of str): List of predicted strings.

    Returns:
        float: The average Jaccard score.
    """
    if not ground_truths or not predictions:
        return 0.0

    if len(ground_truths) != len(predictions):
        raise ValueError("Length of ground truths and predictions must match.")

    scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)
