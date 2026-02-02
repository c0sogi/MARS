import os
import random
import numpy as np
import torch
import scipy.ndimage
from itertools import groupby
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences (lists of integers).

    Args:
        seq1 (list): First sequence of gesture IDs.
        seq2 (list): Second sequence of gesture IDs.

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
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return int(matrix[size_x - 1, size_y - 1])


def median_filter(predictions, window_size=5):
    """
    Applies a median filter to smooth the frame-wise predictions.

    Args:
        predictions (np.ndarray): 1D array of class indices.
        window_size (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply filter using nearest padding to handle edges
    return scipy.ndimage.median_filter(predictions, size=window_size, mode="nearest")


def rle_decode(
    predictions, background_class_id=Config.BACKGROUND_CLASS_ID, min_duration=5
):
    """
    Decodes frame-wise predictions into an ordered list of gestures using Run-Length Encoding logic.
    Filters out background class and segments shorter than min_duration.

    Args:
        predictions (np.ndarray or list): 1D array of class indices.
        background_class_id (int): ID of the background class to ignore.
        min_duration (int): Minimum number of consecutive frames to consider a valid gesture instance.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    gesture_sequence = []

    # Group consecutive identical values
    for key, group in groupby(predictions):
        # Convert key to int just in case
        key_int = int(key)

        # Calculate length of the segment
        length = sum(1 for _ in group)

        # Skip background
        if key_int == background_class_id:
            continue

        # Skip short segments
        if length < min_duration:
            continue

        gesture_sequence.append(key_int)

    return gesture_sequence
