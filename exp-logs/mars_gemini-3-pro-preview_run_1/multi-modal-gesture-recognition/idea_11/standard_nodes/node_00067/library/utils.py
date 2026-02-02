import os
import random
import numpy as np
import torch
import scipy.ndimage
import nltk
from itertools import groupby
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_median_filter(probs, kernel_size=Config.MEDIAN_FILTER_KERNEL):
    """
    Applies a median filter to smooth frame-wise probability maps along the temporal axis.

    Args:
        probs (np.ndarray or torch.Tensor): Input probabilities of shape (Time, Classes).
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed probabilities.
    """
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()

    # Apply median filter along the temporal axis (axis 0)
    # size=(kernel_size, 1) ensures we filter temporally for each class independently
    # Mode 'nearest' repeats the edge values, which is appropriate for time sequences
    smoothed_probs = scipy.ndimage.median_filter(
        probs, size=(kernel_size, 1), mode="nearest"
    )
    return smoothed_probs


def decode_predictions_rle(frame_labels, min_length=Config.MIN_GESTURE_LENGTH):
    """
    Decodes frame-wise labels into an ordered list of gesture IDs using Run-Length Encoding.
    Filters out background class (0) and segments shorter than min_length.

    Args:
        frame_labels (np.ndarray or list): Sequence of class IDs.
        min_length (int): Minimum duration (in frames) for a gesture to be considered valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_labels) == 0:
        return []

    predicted_gestures = []

    # Group consecutive identical values
    for label, group in groupby(frame_labels):
        label = int(label)
        length = sum(1 for _ in group)

        # Filter out background class (0 is reserved for background)
        if label == 0:
            continue

        # Filter out short segments (noise)
        if length < min_length:
            continue

        predicted_gestures.append(label)

    return predicted_gestures


def post_process_sequence(
    probs, kernel_size=Config.MEDIAN_FILTER_KERNEL, min_length=Config.MIN_GESTURE_LENGTH
):
    """
    Full post-processing pipeline: Smooth Probs -> Argmax -> RLE Decode.

    Args:
        probs (np.ndarray or torch.Tensor): Frame-wise probabilities (Time, Classes).
        kernel_size (int): Window size for median smoothing.
        min_length (int): Minimum frame length for valid gestures.

    Returns:
        list: Final list of predicted gesture IDs.
    """
    # 1. Smooth probabilities
    smoothed = apply_median_filter(probs, kernel_size)

    # 2. Get frame-wise class labels
    frame_labels = np.argmax(smoothed, axis=1)

    # 3. Decode to gesture sequence
    gesture_ids = decode_predictions_rle(frame_labels, min_length)

    return gesture_ids


def get_levenshtein_distance(pred_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        pred_seq (list): List of predicted gesture IDs.
        target_seq (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(pred_seq, target_seq)


def compute_levenshtein_score(all_preds, all_targets):
    """
    Computes the global Levenshtein Error Rate.
    Metric = Sum(Levenshtein Distances) / Total Ground Truth Gestures.

    Args:
        all_preds (list of lists): Predicted sequences for the dataset.
        all_targets (list of lists): Ground truth sequences for the dataset.

    Returns:
        float: The computed error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for p, t in zip(all_preds, all_targets):
        dist = get_levenshtein_distance(p, t)
        total_distance += dist
        total_gestures += len(t)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
