import os
import random
import numpy as np
import torch
import nltk
from library.config import MIN_GESTURE_DURATION, BACKGROUND_CLASS_ID


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
    os.environ["PYTHONHASHSEED"] = str(seed)


def levenshtein_distance(hypothesis, reference):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.
    This is a wrapper around nltk.edit_distance.

    Args:
        hypothesis (list): List of predicted gesture IDs.
        reference (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(hypothesis, reference)


def filter_short_segments(rle_sequence, min_duration=MIN_GESTURE_DURATION):
    """
    Filters out gesture segments that are shorter than the minimum duration.
    Background segments are ignored (not added to the result list).

    Args:
        rle_sequence (list): List of (label, count) tuples from RLE.
        min_duration (int): Minimum number of frames required to keep a gesture.

    Returns:
        list: Ordered list of gesture IDs (excluding background).
    """
    filtered_gestures = []
    for label, count in rle_sequence:
        # Skip background class
        if label == BACKGROUND_CLASS_ID:
            continue

        # Keep gesture if it meets the duration threshold
        if count >= min_duration:
            filtered_gestures.append(int(label))

    return filtered_gestures


def decode_predictions(probabilities, min_duration=MIN_GESTURE_DURATION):
    """
    Decodes frame-wise probabilities into a sequence of gesture labels.

    Process:
    1. Applies Argmax to probabilities to get frame labels.
    2. Performs Run-Length Encoding (RLE) to group consecutive frames.
    3. Executes filter_short_segments to remove noise and short predictions.

    Args:
        probabilities (np.ndarray or torch.Tensor): Shape (T, NumClasses) or (T,).
        min_duration (int): Minimum duration in frames.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # Ensure input is numpy array
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    # Step 1: Argmax
    if probabilities.ndim > 1:
        labels = np.argmax(probabilities, axis=-1)
    else:
        labels = probabilities

    if len(labels) == 0:
        return []

    # Step 2: Run-Length Encoding
    rle = []
    curr_val = labels[0]
    curr_count = 1

    for i in range(1, len(labels)):
        if labels[i] == curr_val:
            curr_count += 1
        else:
            rle.append((curr_val, curr_count))
            curr_val = labels[i]
            curr_count = 1
    rle.append((curr_val, curr_count))

    # Step 3: Filter
    return filter_short_segments(rle, min_duration)


def compute_challenge_metric(predictions, ground_truths):
    """
    Computes the competition metric: Sum of Levenshtein distances divided by
    the total number of gestures in the ground truth.

    Args:
        predictions (list of lists): Predicted gesture sequences.
        ground_truths (list of lists): Ground truth gesture sequences.

    Returns:
        float: The error rate (Levenshtein Distance / Total True Gestures).
    """
    total_distance = 0
    total_ref_gestures = 0

    for hyp, ref in zip(predictions, ground_truths):
        dist = levenshtein_distance(hyp, ref)
        total_distance += dist
        total_ref_gestures += len(ref)

    if total_ref_gestures == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_ref_gestures
