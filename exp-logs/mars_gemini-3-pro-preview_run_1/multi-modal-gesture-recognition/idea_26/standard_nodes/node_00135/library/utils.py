import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein distance between two sequences (lists of integers).

    Args:
        s1 (list): First sequence.
        s2 (list): Second sequence.

    Returns:
        int: The edit distance.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_levenshtein(predictions, ground_truths):
    """
    Computes the normalized Levenshtein distance (Error Rate) for a batch of sequences.

    Metric = (Sum of Levenshtein Distances) / (Total Number of Gestures in Ground Truth)

    Args:
        predictions (list of list of int): Predicted gesture sequences (class IDs).
        ground_truths (list of list of int): Ground truth gesture sequences (class IDs).

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        # Ensure inputs are lists and handle potential None types
        p = list(pred) if pred is not None else []
        t = list(truth) if truth is not None else []

        # Compute distance for this pair
        dist = _levenshtein_distance(p, t)

        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length
