import os
import random
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


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
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences of integers.

    Args:
        seq1 (list or np.ndarray): First sequence (e.g., predicted gestures).
        seq2 (list or np.ndarray): Second sequence (e.g., ground truth gestures).

    Returns:
        int: The Levenshtein distance.
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


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the normalized Levenshtein distance (Error Rate) for a batch of sequences.
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths).

    Args:
        predictions (list of lists): List of predicted gesture sequences.
        ground_truths (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The normalized error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        dist = compute_levenshtein(pred, truth)
        total_distance += dist
        total_length += len(truth)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def smooth_predictions(predictions, window_size=7):
    """
    Applies a median filter with nearest-neighbor padding to smooth frame-wise predictions.

    Args:
        predictions (np.ndarray): 1D array of class indices.
        window_size (int): Size of the median filter window (should be odd).

    Returns:
        np.ndarray: Smoothed predictions.
    """
    predictions = np.array(predictions)
    if window_size <= 1 or len(predictions) == 0:
        return predictions

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    pad_width = window_size // 2

    # Nearest-neighbor padding (edge padding) to preserve boundary information
    padded = np.pad(predictions, pad_width, mode="edge")

    # Apply median filter using sliding window view
    windows = sliding_window_view(padded, window_shape=window_size)

    # Compute median along the window axis
    smoothed = np.median(windows, axis=1)

    # Cast back to original type (usually int for classes)
    return smoothed.astype(predictions.dtype)


def decode_predictions(frame_predictions, background_class=0):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs.
    Collapses consecutive duplicates and removes the background class.

    Args:
        frame_predictions (np.ndarray or list): Sequence of frame labels.
        background_class (int): Label ID for the background/null class.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Collapse duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # Remove background
    decoded = [label for label in collapsed if label != background_class]

    return decoded
