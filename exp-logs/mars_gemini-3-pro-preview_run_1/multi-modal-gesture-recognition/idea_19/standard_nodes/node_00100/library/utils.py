import os
import random
import numpy as np
import torch
import scipy.signal


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.

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


def median_filter_1d(signal, kernel_size=5):
    """
    Applies a 1D median filter to the signal.

    Args:
        signal (np.array): 1D array of class indices.
        kernel_size (int): Size of the window. Must be odd.

    Returns:
        np.array: Smoothed signal.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Use scipy.signal.medfilt for efficient computation
    # Ensure signal is at least the size of the kernel to avoid errors,
    # though medfilt handles boundaries by zero-padding which might not be ideal for classification.
    # A safer approach for classification indices is manual or careful padding,
    # but medfilt is standard.
    if len(signal) < kernel_size:
        return signal

    return scipy.signal.medfilt(signal, kernel_size=kernel_size).astype(int)


def decode_predictions(frame_probs, background_id=0, min_len=5, median_filter_size=5):
    """
    Decodes frame-wise probabilities into a sequence of gesture labels using RLE and filtering.

    Args:
        frame_probs (np.array): Shape (T, C) or (T,) containing probabilities or class indices.
        background_id (int): The ID representing the background class.
        min_len (int): Minimum duration (in frames) for a gesture to be kept.
        median_filter_size (int): Window size for smoothing.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # 1. Convert probabilities to class indices if necessary
    if frame_probs.ndim == 2:
        preds = np.argmax(frame_probs, axis=1)
    else:
        preds = frame_probs.copy()

    # 2. Apply Median Filter to smooth noise
    if median_filter_size > 1:
        preds = median_filter_1d(preds, kernel_size=median_filter_size)

    # 3. Run-Length Encoding (RLE)
    # Identify segments: [(label, length), ...]
    segments = []
    if len(preds) == 0:
        return []

    current_label = preds[0]
    current_len = 1

    for i in range(1, len(preds)):
        if preds[i] == current_label:
            current_len += 1
        else:
            segments.append((current_label, current_len))
            current_label = preds[i]
            current_len = 1
    segments.append((current_label, current_len))

    # 4. Filter segments
    final_sequence = []
    for label, length in segments:
        # Skip background class
        if label == background_id:
            continue

        # Skip short segments
        if length < min_len:
            continue

        final_sequence.append(int(label))

    return final_sequence
