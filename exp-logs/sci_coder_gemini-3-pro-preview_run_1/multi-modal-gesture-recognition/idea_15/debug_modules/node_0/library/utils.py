import os
import random
import numpy as np
import torch
from scipy.signal import medfilt
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.

    Args:
        seq1 (list[int]): First sequence.
        seq2 (list[int]): Second sequence.

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


def compute_error_rate(predictions, targets):
    """
    Computes the normalized Levenshtein error rate for a batch of predictions.
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths)

    Args:
        predictions (list[list[int]]): List of predicted gesture sequences.
        targets (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        dist = levenshtein_distance(pred, target)
        total_distance += dist
        total_length += len(target)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def rle_decode(frame_predictions):
    """
    Decodes frame-wise class predictions into a sequence of gesture labels
    using Run-Length Encoding (RLE) and Median Filtering.

    1. Applies median filter to smooth noise.
    2. Groups contiguous frames with the same label.
    3. Filters out the Background class (Config.BACKGROUND_CLASS_ID).
    4. Filters out segments shorter than Config.MIN_GESTURE_LENGTH.

    Args:
        frame_predictions (np.ndarray or list): 1D array of frame-wise class IDs.

    Returns:
        list[int]: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Ensure input is numpy array
    preds = np.array(frame_predictions, dtype=int)

    # 1. Apply Median Filter
    # Kernel size must be odd
    kernel_size = Config.MEDIAN_FILTER_KERNEL
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed_preds = medfilt(preds, kernel_size=kernel_size).astype(int)

    # 2. Run-Length Encoding
    decoded_sequence = []

    if len(smoothed_preds) == 0:
        return decoded_sequence

    # Identify changes in value
    # Append a dummy value at the end to capture the last segment
    padded_preds = np.concatenate([smoothed_preds, [-1]])

    current_label = padded_preds[0]
    current_len = 0

    for label in padded_preds:
        if label == current_label:
            current_len += 1
        else:
            # Segment ended, process it
            # 3. Filter Background
            if current_label != Config.BACKGROUND_CLASS_ID:
                # 4. Filter Short Segments
                if current_len >= Config.MIN_GESTURE_LENGTH:
                    decoded_sequence.append(current_label)

            # Reset for next segment
            current_label = label
            current_len = 1

    return decoded_sequence
