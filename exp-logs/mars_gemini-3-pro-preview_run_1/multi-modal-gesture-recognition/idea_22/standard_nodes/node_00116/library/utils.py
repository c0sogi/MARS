import os
import random
import numpy as np
import torch
from scipy.signal import medfilt
from library.config import Config


def set_seed():
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config.set_seed method.
    """
    Config.set_seed()


def levenshtein_distance(p, t):
    """
    Calculates the Levenshtein distance between two sequences.

    Args:
        p (list): Predicted sequence of labels.
        t (list): Target (ground truth) sequence of labels.

    Returns:
        int: The edit distance.
    """
    m, n = len(p), len(t)
    if m == 0:
        return n
    if n == 0:
        return m

    # Initialize matrix
    d = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == t[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,
            )  # Substitution

    return d[m, n]


def compute_levenshtein_score(predictions, targets):
    """
    Computes the aggregate Levenshtein error rate for a batch or dataset.
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths)

    Args:
        predictions (list of lists): List of predicted label sequences.
        targets (list of lists): List of ground truth label sequences.

    Returns:
        float: The error rate.
    """
    total_distance = 0
    total_length = 0

    for p, t in zip(predictions, targets):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def median_filter(predictions, window_size=None):
    """
    Applies a median filter to smooth frame-wise predictions.

    Args:
        predictions (np.ndarray or list): Frame-wise class indices.
        window_size (int, optional): Size of the smoothing window.
                                     Defaults to Config.MEDIAN_FILTER_WINDOW.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    if window_size is None:
        window_size = Config.MEDIAN_FILTER_WINDOW

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    return medfilt(predictions, kernel_size=window_size).astype(int)


def rle_decode(predictions, background_id=None, min_length=None):
    """
    Decodes frame-wise predictions into an ordered sequence of gesture labels
    using Run-Length Encoding logic. Filters out background and short segments.

    Args:
        predictions (np.ndarray or list): Frame-wise class indices (smoothed).
        background_id (int, optional): The class ID representing background.
                                       Defaults to Config.BACKGROUND_CLASS_ID.
        min_length (int, optional): Minimum duration (frames) to keep a gesture.
                                    Defaults to Config.MIN_SEGMENT_LENGTH.

    Returns:
        list: Ordered list of detected gesture IDs.
    """
    if background_id is None:
        background_id = Config.BACKGROUND_CLASS_ID
    if min_length is None:
        min_length = Config.MIN_SEGMENT_LENGTH

    if len(predictions) == 0:
        return []

    # Run-Length Encoding
    decoded_sequence = []

    # Identify segments: (label, length)
    # Using itertools.groupby logic manually for clarity/numpy compatibility
    current_label = predictions[0]
    current_len = 1

    segments = []

    for i in range(1, len(predictions)):
        label = predictions[i]
        if label == current_label:
            current_len += 1
        else:
            segments.append((current_label, current_len))
            current_label = label
            current_len = 1
    segments.append((current_label, current_len))

    # Filter segments
    for label, length in segments:
        if label == background_id:
            continue
        if length >= min_length:
            decoded_sequence.append(int(label))

    return decoded_sequence
