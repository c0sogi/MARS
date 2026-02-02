import os
import random
import numpy as np
import torch
import nltk
from itertools import groupby
from library.config import SEED, LABEL_MAP


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic algorithms are used where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def median_filter_1d(x, k=5):
    """
    Applies a 1D median filter to the input array x with window size k.
    Implemented using NumPy to ensure compatibility without relying on scipy.
    """
    if k % 2 == 0:
        k += 1  # Ensure odd window size

    pad = k // 2
    # Pad with edge values to maintain output size
    x_padded = np.pad(x, (pad, pad), mode="edge")

    # Create sliding windows efficiently
    # shape: (len(x), k)
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, k)

    # Compute median along the window axis
    return np.median(windows, axis=1).astype(x.dtype)


def decode_predictions(frame_logits):
    """
    Decodes frame-wise logits or predictions into a sequence of gesture IDs.
    Applies Median Filtering and Run-Length Encoding with specific filtering logic.

    Args:
        frame_logits (np.ndarray): Shape [T, num_classes] (logits) or [T] (class indices).

    Returns:
        list: Ordered list of gesture IDs (integers).
    """
    # Convert logits to class indices if necessary
    if frame_logits.ndim == 2:
        frame_preds = np.argmax(frame_logits, axis=1)
    else:
        frame_preds = frame_logits

    # 1. Apply Median Filter (window size 5) to smooth noise
    smoothed_preds = median_filter_1d(frame_preds, k=5)

    # 2. Run-Length Encoding & Filtering
    gestures = []

    # Group consecutive identical values
    for label, group in groupby(smoothed_preds):
        # Calculate duration of the segment
        length = sum(1 for _ in group)

        # Filter logic:
        # - Remove Background (class 0)
        # - Remove short segments (duration < 5 frames)
        if label != LABEL_MAP["background"] and length >= 5:
            gestures.append(int(label))

    return gestures


def compute_levenshtein(predicted_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        predicted_seq (list): List of predicted gesture IDs.
        target_seq (list): List of ground truth gesture IDs.

    Returns:
        int: Edit distance.
    """
    return nltk.edit_distance(predicted_seq, target_seq)


def compute_dataset_metric(predictions_list, targets_list):
    """
    Computes the global metric: Total Levenshtein Distance / Total Ground Truth Gestures.
    This score is analogous to an error rate.

    Args:
        predictions_list (list of lists): List of predicted sequences.
        targets_list (list of lists): List of ground truth sequences.

    Returns:
        float: The normalized error rate.
    """
    total_distance = 0
    total_gestures = 0

    for pred, target in zip(predictions_list, targets_list):
        dist = compute_levenshtein(pred, target)
        total_distance += dist
        total_gestures += len(target)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
