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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences using Dynamic Programming.

    Args:
        seq1 (list): First sequence of items (e.g., predicted gesture IDs).
        seq2 (list): Second sequence of items (e.g., target gesture IDs).

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

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
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return int(matrix[size_x - 1, size_y - 1])


def compute_metric_score(predictions, targets):
    """
    Computes the global Levenshtein score (Error Rate) for a batch or dataset.
    Metric = Sum(Levenshtein Distances) / Sum(Total Ground Truth Gestures)

    Args:
        predictions (list of lists): Predicted gesture sequences.
        targets (list of lists): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_gestures = 0

    for pred, targ in zip(predictions, targets):
        total_distance += levenshtein_distance(pred, targ)
        total_gestures += len(targ)

    if total_gestures == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_gestures


def decode_sequence(frame_predictions):
    """
    Decodes frame-wise class predictions into a sequence of gesture IDs.
    Logic:
    1. Collapse consecutive repeated labels (e.g., [1, 1, 2, 2] -> [1, 2]).
    2. Remove background class (ID 0).

    Args:
        frame_predictions (list or np.array): List of frame-wise class indices.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # 1. Collapse consecutive duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # 2. Remove background (assuming background ID is 0)
    final_sequence = [int(x) for x in collapsed if x != 0]

    return final_sequence


def format_submission_line(sample_id, predicted_sequence):
    """
    Formats a single prediction for the submission CSV.
    Format: SessionID,Label1,Label2,...

    Args:
        sample_id (str): The sequence identifier (e.g., 'Session00001').
        predicted_sequence (list): List of gesture IDs.

    Returns:
        str: Formatted string.
    """
    labels_str = ",".join(map(str, predicted_sequence))
    return f"{sample_id},{labels_str}"
