import os
import random
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(p, t):
    """
    Computes the Levenshtein distance between two sequences using dynamic programming.

    Args:
        p (list or np.array): The predicted sequence of gesture IDs.
        t (list or np.array): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    m = len(p)
    n = len(t)

    # Initialize the distance matrix
    d = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j

    # Fill the matrix
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if p[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1

            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,
            )  # Substitution

    return d[m, n]


def compute_error_rate(predictions, ground_truths):
    """
    Computes the normalized Levenshtein Error Rate (LER) for a batch of sequences.

    Metric Definition:
        Sum of Levenshtein distances for all sequences divided by the total number
        of gestures in the ground truth.

    Args:
        predictions (list of lists): List of predicted gesture sequences.
        ground_truths (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The computed error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        dist = levenshtein_distance(pred, truth)
        total_distance += dist
        total_length += len(truth)

    if total_length == 0:
        return 0.0

    return total_distance / total_length
