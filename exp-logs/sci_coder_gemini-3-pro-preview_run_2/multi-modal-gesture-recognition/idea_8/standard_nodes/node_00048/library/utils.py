import random
import numpy as np
import torch
import nltk
import scipy.ndimage
from itertools import groupby


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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


def get_device():
    """
    Returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein distance metric as defined in the task description.

    Metric = (Sum of Levenshtein distances) / (Total number of ground truth gestures)

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        targets (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The computed error rate (lower is better).
    """
    total_distance = 0
    total_target_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure inputs are lists and handle None
        p = list(pred_seq) if pred_seq is not None else []
        t = list(target_seq) if target_seq is not None else []

        # Calculate Levenshtein distance between two sequences
        # nltk.edit_distance works efficiently on lists of integers
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_target_length += len(t)

    if total_target_length == 0:
        return 0.0

    return total_distance / total_target_length


def smooth_predictions(probs, window_size=5):
    """
    Applies Median Filter to probability sequences with nearest-neighbor padding.
    Cite Lesson 6 (Temporal Smoothing) and Lesson 9 (Boundary-Aware Padding).

    Args:
        probs (np.ndarray): Shape [Time, NumClasses]
        window_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Smoothed probabilities.
    """
    # Apply median filter along the time axis (axis 0)
    # size=(window_size, 1) means filter over time, independent over classes
    # mode='nearest' replicates the boundary values
    smoothed_probs = scipy.ndimage.median_filter(
        probs, size=(window_size, 1), mode="nearest"
    )
    return smoothed_probs


def decode_sequence(probs):
    """
    Decodes frame-wise probabilities into a sequence of gesture IDs.
    Strategy: Argmax -> Collapse Repeats -> Remove Background.

    Args:
        probs (np.ndarray): Shape [Time, NumClasses]

    Returns:
        list: Ordered list of gesture IDs (int).
    """
    # 1. Greedy Decoding (Argmax)
    labels = np.argmax(probs, axis=1)

    # 2. Collapse consecutive repeats
    collapsed_labels = [k for k, g in groupby(labels)]

    # 3. Remove Background (Class 0)
    final_sequence = [lbl for lbl in collapsed_labels if lbl != 0]

    return final_sequence


def decode_target_sequence(labels):
    """
    Decodes frame-wise target indices into a sequence of gesture IDs.

    Args:
        labels (array-like): Sequence of class indices.

    Returns:
        list: Ordered list of gesture IDs.
    """
    # Collapse repeats
    collapsed = [k for k, g in groupby(labels)]
    # Remove background
    return [k for k in collapsed if k != 0]
