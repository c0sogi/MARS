import os
import random
import numpy as np
import torch
from itertools import groupby
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


def levenshtein_distance(p, t):
    """
    Calculates the Levenshtein distance between two sequences p (prediction) and t (target).

    Args:
        p (list): Predicted sequence of gesture IDs.
        t (list): Target sequence of gesture IDs.

    Returns:
        int: The Levenshtein distance.
    """
    n = len(p)
    m = len(t)

    if n == 0:
        return m
    if m == 0:
        return n

    # Initialize matrix
    d = np.zeros((n + 1, m + 1), dtype=int)

    # Initialize first row and column
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    # Compute distances
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if p[i - 1] == t[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost,  # Substitution
            )

    return d[n, m]


def compute_levenshtein_ratio(predictions, targets):
    """
    Computes the competition metric: Sum(Levenshtein Distances) / Total True Gestures.

    Args:
        predictions (list of list of int): Predicted sequences of gesture IDs.
        targets (list of list of int): Ground truth sequences of gesture IDs.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_truth_length = 0

    for p, t in zip(predictions, targets):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_truth_length += len(t)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length


def median_filter(predictions, window_size=Config.MEDIAN_FILTER_WINDOW):
    """
    Applies a median filter to the frame-wise predictions to smooth noise.

    Args:
        predictions (np.array or list): Frame-wise class predictions.
        window_size (int): Size of the median filter window.

    Returns:
        np.array: Smoothed predictions.
    """
    # Ensure input is numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    else:
        predictions = np.array(predictions)

    if len(predictions) < window_size:
        return predictions

    # Pad to keep size same (edge padding)
    pad_size = window_size // 2
    padded = np.pad(predictions, (pad_size, pad_size), mode="edge")

    # Create sliding window view using stride_tricks
    shape = (predictions.shape[0], window_size)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Compute median
    smoothed = np.median(windows, axis=1).astype(int)
    return smoothed


def rle_decode(
    frame_predictions,
    background_label=Config.BACKGROUND_LABEL,
    min_len=Config.MIN_SEGMENT_LENGTH,
):
    """
    Decodes frame-wise predictions into an ordered list of gestures using Run-Length Encoding.
    Filters out background class and segments shorter than min_len.

    Args:
        frame_predictions (np.array or list): Sequence of frame-wise class IDs.
        background_label (int): ID of the background class to exclude.
        min_len (int): Minimum length of a segment to be considered a valid gesture.

    Returns:
        list of int: Ordered list of recognized gesture IDs.
    """
    # Normalize input
    if isinstance(frame_predictions, torch.Tensor):
        frame_predictions = frame_predictions.detach().cpu().numpy()
    elif not isinstance(frame_predictions, np.ndarray):
        frame_predictions = np.array(frame_predictions)

    if len(frame_predictions) == 0:
        return []

    decoded_gestures = []

    # Group consecutive identical labels
    for label, group in groupby(frame_predictions):
        # Convert group to list to get length
        length = sum(1 for _ in group)

        # Filter background
        if label == background_label:
            continue

        # Filter short segments
        if length < min_len:
            continue

        decoded_gestures.append(int(label))

    return decoded_gestures
