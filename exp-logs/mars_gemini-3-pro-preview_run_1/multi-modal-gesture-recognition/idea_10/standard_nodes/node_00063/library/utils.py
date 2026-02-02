import os
import random
import numpy as np
import torch
from scipy.ndimage import median_filter as scipy_median_filter
from library.config import SEED, BACKGROUND_CLASS_ID


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        hyp (list[int]): The hypothesis sequence of gesture IDs.
        ref (list[int]): The reference (ground truth) sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    n = len(hyp)
    m = len(ref)

    # Optimization for empty sequences
    if n == 0:
        return m
    if m == 0:
        return n

    # Initialize the previous row of the DP matrix
    # dp[i][j] corresponds to distance between hyp[:i] and ref[:j]
    # We only store two rows for memory efficiency
    prev_row = list(range(m + 1))

    for i, h_val in enumerate(hyp):
        curr_row = [i + 1]
        for j, r_val in enumerate(ref):
            # Substitution cost is 0 if match, 1 if mismatch
            cost = 0 if h_val == r_val else 1

            # Compute minimum of deletion, insertion, substitution
            dist = min(
                prev_row[j + 1] + 1,  # Deletion (from hyp)
                curr_row[j] + 1,  # Insertion (into hyp)
                prev_row[j] + cost,  # Substitution
            )
            curr_row.append(dist)
        prev_row = curr_row

    return prev_row[-1]


def median_filter(data, window_size=5):
    """
    Applies a median filter to smooth data along the time dimension.
    Supports both 1D arrays (labels) and 2D arrays (probability maps).

    Args:
        data (np.ndarray): Input data of shape (T,) or (T, C).
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: The smoothed data.
    """
    data = np.array(data)

    if data.ndim == 1:
        # 1D case: Smoothing labels or single-channel signal
        return scipy_median_filter(data, size=window_size, mode="nearest")
    elif data.ndim == 2:
        # 2D case: Smoothing probability maps (Time x Channels)
        # We filter along axis 0 (Time) with a kernel of (window_size, 1)
        return scipy_median_filter(data, size=(window_size, 1), mode="nearest")
    else:
        raise ValueError(f"Input data must be 1D or 2D, got shape {data.shape}")


def rle_decode(predictions, background_id=BACKGROUND_CLASS_ID, min_duration=5):
    """
    Decodes a sequence of frame-wise predictions into a list of gesture IDs.
    Applies Run-Length Encoding logic to group consecutive predictions,
    then filters out background classes and segments shorter than min_duration.

    Args:
        predictions (list or np.ndarray): Frame-wise class predictions.
        background_id (int): The class ID to be treated as background/noise.
        min_duration (int): Minimum frame duration to accept a gesture segment.

    Returns:
        list[int]: The ordered sequence of detected gesture IDs.
    """
    if len(predictions) == 0:
        return []

    predictions = np.array(predictions)

    # Find indices where the value changes
    # predictions[:-1] != predictions[1:] gives a boolean array
    # np.where returns indices, we add 1 to get the start of the new segment
    change_indices = np.where(predictions[:-1] != predictions[1:])[0] + 1

    # Define segment boundaries: [0, change_1, change_2, ..., length]
    boundaries = np.concatenate(([0], change_indices, [len(predictions)]))

    result = []

    # Iterate over segments
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Get the label for this segment
        label = predictions[start]
        duration = end - start

        # Filter logic
        if label != background_id and duration >= min_duration:
            result.append(int(label))

    return result
