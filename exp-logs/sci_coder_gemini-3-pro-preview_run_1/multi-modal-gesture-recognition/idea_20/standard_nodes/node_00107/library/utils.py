import numpy as np
import torch
import scipy.signal
from library.config import Config


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        seq1 (list): First sequence (e.g., prediction).
        seq2 (list): Second sequence (e.g., ground truth).

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
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the competition metric: Sum of Levenshtein distances / Total ground truth gestures.

    Args:
        predictions (list of lists): Predicted gesture sequences.
        ground_truths (list of lists): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_len = 0

    for pred, truth in zip(predictions, ground_truths):
        dist = levenshtein_distance(pred, truth)
        total_distance += dist
        total_len += len(truth)

    if total_len == 0:
        return 0.0

    return total_distance / total_len


def apply_median_filter(predictions, kernel_size=5):
    """
    Applies a median filter to smooth frame-wise predictions.

    Args:
        predictions (np.ndarray or torch.Tensor): 1D array of frame labels.
        kernel_size (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    return scipy.signal.medfilt(predictions, kernel_size=kernel_size).astype(int)


def rle_decode(
    frame_predictions, min_duration=5, background_id=Config.BACKGROUND_CLASS_ID
):
    """
    Decodes frame-wise predictions into an ordered list of gestures using Run-Length Encoding.
    Filters out background class and short segments.

    Args:
        frame_predictions (np.ndarray or list): Sequence of frame labels.
        min_duration (int): Minimum number of frames to consider a valid gesture instance.
        background_id (int): The class ID representing background/silence.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # Run-Length Encoding
    collapsed_sequence = []

    # Find changes in the sequence
    # Append a dummy value at the end to capture the last segment
    padded_preds = np.concatenate([frame_predictions, [-1]])

    # Where does the value change?
    change_indices = np.where(padded_preds[:-1] != padded_preds[1:])[0]

    start_idx = 0
    for end_idx in change_indices:
        # end_idx is inclusive in the segment
        segment_len = (end_idx - start_idx) + 1
        label = frame_predictions[start_idx]

        # Filter logic
        if label != background_id and segment_len >= min_duration:
            collapsed_sequence.append(int(label))

        start_idx = end_idx + 1

    return collapsed_sequence


def predictions_to_submission_format(sample_ids, gesture_sequences):
    """
    Formats predictions for submission.

    Args:
        sample_ids (list): List of sample IDs (e.g., 'Sample00001').
        gesture_sequences (list of lists): List of predicted gesture IDs.

    Returns:
        list of str: Lines formatted as 'SequenceID,Label1,Label2,...'
    """
    lines = []
    for sid, seq in zip(sample_ids, gesture_sequences):
        seq_str = ",".join(map(str, seq))
        lines.append(f"{sid},{seq_str}")
    return lines
