import os
import random
import numpy as np
import torch
import scipy.signal
from nltk import edit_distance
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_decode(frame_probs, min_length=Config.MIN_SEGMENT_LENGTH, background_class=0):
    """
    Decodes frame-wise probabilities into a list of gesture IDs using
    Run-Length Encoding (RLE) with median filtering and length thresholding.

    Args:
        frame_probs (np.ndarray): Shape (T, NumClasses) or (T,) containing probabilities or class indices.
        min_length (int): Minimum number of frames to consider a valid segment.
        background_class (int): The class ID representing background/no-gesture.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # Convert to class indices if input is probability matrix
    if frame_probs.ndim == 2:
        labels = np.argmax(frame_probs, axis=1)
    else:
        labels = frame_probs

    # Apply Median Filter to smooth predictions (removes jitter)
    # Kernel size must be odd
    kernel_size = Config.MEDIAN_FILTER_KERNEL
    if kernel_size % 2 == 0:
        kernel_size += 1

    smoothed_labels = scipy.signal.medfilt(labels, kernel_size=kernel_size).astype(int)

    # Run-Length Encoding
    predicted_gestures = []

    if len(smoothed_labels) == 0:
        return predicted_gestures

    # Iteration to group consecutive labels
    current_label = smoothed_labels[0]
    current_count = 1

    for i in range(1, len(smoothed_labels)):
        lbl = smoothed_labels[i]
        if lbl == current_label:
            current_count += 1
        else:
            # End of a segment
            if current_label != background_class and current_count >= min_length:
                predicted_gestures.append(int(current_label))

            # Start new segment
            current_label = lbl
            current_count = 1

    # Handle the last segment
    if current_label != background_class and current_count >= min_length:
        predicted_gestures.append(int(current_label))

    return predicted_gestures


def compute_levenshtein_ratio(predictions, targets):
    """
    Computes the Levenshtein Error Rate: Total Edit Distance / Total Ground Truth Gestures.

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        targets (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_truth_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Calculate Levenshtein distance for this sequence
        dist = edit_distance(pred_seq, target_seq)
        total_distance += dist
        total_truth_length += len(target_seq)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length
