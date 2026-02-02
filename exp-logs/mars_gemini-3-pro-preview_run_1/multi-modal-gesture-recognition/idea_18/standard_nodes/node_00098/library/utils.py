import os
import random
import numpy as np
import torch
import nltk
from scipy.ndimage import median_filter as scipy_median_filter
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein_ratio(predicted_sequences, target_sequences):
    """
    Computes the Levenshtein error rate metric.

    Metric = (Sum of Levenshtein distances) / (Total number of gestures in ground truth)

    Args:
        predicted_sequences (list of list of int): List of predicted gesture ID sequences.
        target_sequences (list of list of int): List of ground truth gesture ID sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_target_length = 0

    # Ensure inputs are lists of lists
    if len(predicted_sequences) != len(target_sequences):
        raise ValueError(
            f"Mismatch in number of sequences: preds={len(predicted_sequences)}, targets={len(target_sequences)}"
        )

    for pred, target in zip(predicted_sequences, target_sequences):
        # Calculate Levenshtein distance for this sequence pair
        dist = nltk.edit_distance(pred, target)
        total_distance += dist
        total_target_length += len(target)

    if total_target_length == 0:
        return 0.0

    return total_distance / total_target_length


def apply_median_filter(predictions, window_size=Config.MEDIAN_FILTER_WINDOW):
    """
    Applies a median filter to smooth frame-wise predictions.

    Args:
        predictions (np.ndarray): 1D array of class indices.
        window_size (int): The size of the filter window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    return scipy_median_filter(predictions, size=window_size, mode="nearest")


def decode_predictions(
    frame_predictions,
    min_segment_length=Config.MIN_SEGMENT_LENGTH,
    background_class_id=Config.BACKGROUND_CLASS_ID,
):
    """
    Decodes frame-wise class predictions into a sequence of gesture labels.

    Steps:
    1. Apply Run-Length Encoding (RLE) to group consecutive identical frames.
    2. Filter out segments shorter than min_segment_length.
    3. Filter out background class segments.

    Args:
        frame_predictions (np.ndarray or list): 1D sequence of frame labels.
        min_segment_length (int): Minimum frames required to consider a segment valid.
        background_class_id (int): The ID representing the background/null class.

    Returns:
        list of int: The ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Run-Length Encoding
    # Identify change points
    frame_predictions = np.array(frame_predictions)
    # Append a value different from the last to ensure the last segment is captured if we used diff
    # Alternatively, manual iteration is robust and simple for RLE

    segments = []
    if len(frame_predictions) > 0:
        current_label = frame_predictions[0]
        current_len = 1

        for label in frame_predictions[1:]:
            if label == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = label
                current_len = 1
        segments.append((current_label, current_len))

    # Filter segments
    final_sequence = []
    for label, length in segments:
        # Check if it's a gesture of interest (not background)
        if label != background_class_id:
            # Check duration constraint
            if length >= min_segment_length:
                final_sequence.append(int(label))

    return final_sequence
