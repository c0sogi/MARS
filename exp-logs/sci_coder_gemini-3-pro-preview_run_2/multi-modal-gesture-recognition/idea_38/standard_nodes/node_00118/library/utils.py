import numpy as np
import nltk
from scipy.ndimage import median_filter
from library.config import Config, set_seed


def compute_levenshtein(predicted_seqs, truth_seqs):
    """
    Computes the Levenshtein distance metric averaged over the total number of ground truth gestures.

    Metric Definition:
    Sum of Levenshtein distances for all sequences divided by the total number of gestures
    in the truth value file. This score is analogous to an error rate and can exceed one.

    Args:
        predicted_seqs (list of list of int): List of predicted gesture sequences (lists of class IDs).
        truth_seqs (list of list of int): List of ground truth gesture sequences (lists of class IDs).

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_truth_length = 0

    for pred, truth in zip(predicted_seqs, truth_seqs):
        # Ensure inputs are lists
        p = list(pred) if pred is not None else []
        t = list(truth) if truth is not None else []

        # Calculate Levenshtein distance
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_truth_length += len(t)

    # Avoid division by zero if the test set has no gestures (unlikely but safe)
    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length


def median_filter_predictions(predictions, kernel_size=None):
    """
    Applies median filtering to frame-wise predictions to smooth noise.
    Uses Nearest-Neighbor Padding to preserve gestures at sequence boundaries.

    Args:
        predictions (np.ndarray): Array of shape (T,) or (B, T) containing class indices.
        kernel_size (int, optional): Size of the median filter window.
                                     Defaults to Config.MEDIAN_FILTER_KERNEL.

    Returns:
        np.ndarray: The filtered predictions with the same shape as input.
    """
    if kernel_size is None:
        kernel_size = Config.MEDIAN_FILTER_KERNEL

    # Ensure numpy array
    preds = np.array(predictions)

    # Apply filter with 'nearest' mode to handle boundaries
    if preds.ndim == 1:
        return median_filter(preds, size=kernel_size, mode="nearest")
    elif preds.ndim == 2:
        filtered_batch = []
        for i in range(preds.shape[0]):
            f = median_filter(preds[i], size=kernel_size, mode="nearest")
            filtered_batch.append(f)
        return np.array(filtered_batch)
    else:
        raise ValueError(f"Predictions must be 1D or 2D array, got shape {preds.shape}")


def decode_predictions(frame_predictions):
    """
    Decodes frame-wise class indices into a sequence of gesture labels.

    Steps:
    1. Collapses consecutive repeated labels.
    2. Removes the background class (0).

    Args:
        frame_predictions (np.ndarray or list): Frame-wise class indices.

    Returns:
        list of int: The ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # Remove background (class 0)
    final_sequence = [x for x in collapsed if x != 0]

    return final_sequence
