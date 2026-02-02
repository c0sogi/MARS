import os
import random
import numpy as np
import torch
import nltk
from scipy.ndimage import median_filter
from itertools import groupby


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smooth_predictions(predictions, window_size: int = 5):
    """
    Applies a median filter to the frame-wise predictions to remove high-frequency jitter.

    Args:
        predictions (np.ndarray or list): 1D array of class indices.
        window_size (int): The size of the median filter window. Must be odd.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Use nearest mode to replicate boundary values, preventing data loss at edges
    smoothed = median_filter(predictions, size=window_size, mode="nearest")
    return smoothed


def decode_sequence(frame_predictions, background_class_id: int = 0):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs.
    Logic: Collapse consecutive repeats -> Remove background class.

    Args:
        frame_predictions (list or np.ndarray): Sequence of frame labels.
        background_class_id (int): The label ID representing the background/no-gesture class.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # Collapse consecutive duplicates
    collapsed = [k for k, g in groupby(frame_predictions)]

    # Remove background class
    decoded = [x for x in collapsed if x != background_class_id]

    return decoded


def compute_levenshtein(prediction, target):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        prediction (list): Predicted sequence of gesture IDs.
        target (list): Ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(prediction, target)


def compute_normalized_levenshtein(predictions_list, targets_list):
    """
    Computes the competition metric: Total Levenshtein Distance / Total True Gestures.

    Args:
        predictions_list (list of lists): List of predicted gesture sequences.
        targets_list (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The normalized error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for pred, targ in zip(predictions_list, targets_list):
        dist = compute_levenshtein(pred, targ)
        total_distance += dist
        total_gestures += len(targ)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
