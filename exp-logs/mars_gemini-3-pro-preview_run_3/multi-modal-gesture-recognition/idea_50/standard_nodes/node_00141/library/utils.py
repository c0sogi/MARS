import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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
    Calculates the Levenshtein distance between two sequences.

    Args:
        seq1 (list or np.array): First sequence of IDs.
        seq2 (list or np.array): Second sequence of IDs.

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
    Computes the normalized Levenshtein distance score for a batch of predictions.
    Score = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths)

    Args:
        predictions (list of lists): Predicted gesture sequences.
        targets (list of lists): Ground truth gesture sequences.

    Returns:
        float: The normalized error rate.
    """
    total_distance = 0
    total_truth_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        dist = levenshtein_distance(pred_seq, target_seq)
        total_distance += dist
        total_truth_length += len(target_seq)

    if total_truth_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_truth_length


def decode_predictions_to_sequence(
    frame_predictions, min_duration=None, background_class=None
):
    """
    Decodes frame-wise class predictions into a sequence of gesture IDs.
    Applies Run-Length Encoding and filters short segments.

    Args:
        frame_predictions (np.array or list): Frame-wise class indices.
        min_duration (int, optional): Minimum frames to keep a segment. Defaults to config.MIN_DURATION.
        background_class (int, optional): ID of the background class to exclude. Defaults to config.BACKGROUND_CLASS_ID.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if min_duration is None:
        min_duration = config.MIN_DURATION
    if background_class is None:
        background_class = config.BACKGROUND_CLASS_ID

    if len(frame_predictions) == 0:
        return []

    # Run-Length Encoding
    segments = []
    if len(frame_predictions) > 0:
        current_label = frame_predictions[0]
        current_len = 1

        for label in frame_predictions[1:]:
            if label == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = label
                current_len = 1
        segments.append((current_label, current_len))

    # Filter and Construct Sequence
    final_sequence = []
    for label, length in segments:
        # Skip background class
        if label == background_class:
            continue

        # Skip segments shorter than min_duration
        if length < min_duration:
            continue

        final_sequence.append(int(label))

    return final_sequence


def run_length_encoding(frame_predictions):
    """
    Raw Run-Length Encoding helper.

    Args:
        frame_predictions (list or np.array): Input sequence.

    Returns:
        list of tuples: [(label, length), ...]
    """
    if len(frame_predictions) == 0:
        return []

    segments = []
    current_label = frame_predictions[0]
    current_len = 1

    for label in frame_predictions[1:]:
        if label == current_label:
            current_len += 1
        else:
            segments.append((current_label, current_len))
            current_label = label
            current_len = 1
    segments.append((current_label, current_len))

    return segments
