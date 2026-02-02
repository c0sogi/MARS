import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
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


def get_device():
    """
    Returns the appropriate torch device (cuda if available, else cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        seq1 (list or np.array): First sequence of items (e.g., predicted gesture IDs).
        seq2 (list or np.array): Second sequence of items (e.g., ground truth gesture IDs).

    Returns:
        int: The edit distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = np.zeros((size_x, size_y), dtype=int)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )

    return matrix[size_x - 1, size_y - 1]


def compute_normalized_levenshtein(predictions, targets):
    """
    Computes the normalized Levenshtein distance metric for the challenge.

    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures

    Args:
        predictions (list of lists): List of predicted gesture sequences.
        targets (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The normalized error score.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have the same length."
        )

    total_distance = 0
    total_truth_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure sequences are lists of integers
        p = [int(x) for x in pred_seq]
        t = [int(x) for x in target_seq]

        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_truth_length += len(t)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length
