import os
import random
import numpy as np
import torch
import scipy.ndimage
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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
    Computes the Levenshtein edit distance between two sequences.

    Args:
        seq1 (list or np.array): First sequence of gesture IDs.
        seq2 (list or np.array): Second sequence of gesture IDs.

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


def median_filter_predictions(predictions, window_size=Config.MEDIAN_FILTER_WINDOW):
    """
    Applies a median filter to smooth frame-wise predictions.

    Args:
        predictions (np.array): 1D array of predicted class indices.
        window_size (int): The size of the sliding window.

    Returns:
        np.array: Smoothed predictions.
    """
    return scipy.ndimage.median_filter(predictions, size=window_size, mode="nearest")


def decode_predictions_to_gestures(
    frame_predictions,
    background_label=Config.BACKGROUND_LABEL,
    min_length=Config.MIN_GESTURE_LENGTH,
):
    """
    Converts frame-wise predictions into a sequence of gesture labels using
    Run-Length Encoding (RLE) logic. Filters out background and short segments.

    Args:
        frame_predictions (np.array): 1D array of frame-wise class indices.
        background_label (int): The class index representing background/silence.
        min_length (int): Minimum number of frames for a gesture to be valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Run-Length Encoding
    # Find indices where values change
    change_indices = np.where(frame_predictions[:-1] != frame_predictions[1:])[0] + 1

    # Add start and end indices
    split_indices = np.concatenate(([0], change_indices, [len(frame_predictions)]))

    gesture_sequence = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        label = frame_predictions[start]
        duration = end - start

        # Filter background and short gestures
        if label != background_label and duration >= min_length:
            gesture_sequence.append(int(label))

    return gesture_sequence
