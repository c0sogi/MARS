import os
import numpy as np
import scipy.signal
from library.config import Config


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        hyp (list[int]): The hypothesis (predicted) sequence of gesture IDs.
        ref (list[int]): The reference (ground truth) sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    n = len(hyp)
    m = len(ref)

    # Initialize DP matrix
    # dist[i][j] is the distance between hyp[:i] and ref[:j]
    dist = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases: transforming empty string to/from non-empty
    for i in range(n + 1):
        dist[i, 0] = i
    for j in range(m + 1):
        dist[0, j] = j

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            dist[i, j] = min(
                dist[i - 1, j] + 1,  # Deletion
                dist[i, j - 1] + 1,  # Insertion
                dist[i - 1, j - 1] + cost,  # Substitution
            )

    return dist[n, m]


def compute_levenshtein_ratio(hypotheses, references):
    """
    Computes the Levenshtein Error Rate for a batch of predictions.
    Metric = Sum(Distances) / Sum(Reference Lengths).

    Args:
        hypotheses (list[list[int]]): List of predicted gesture sequences.
        references (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_dist = 0
    total_ref_len = 0

    for h, r in zip(hypotheses, references):
        total_dist += levenshtein_distance(h, r)
        total_ref_len += len(r)

    # Avoid division by zero
    if total_ref_len == 0:
        return 0.0

    return total_dist / total_ref_len


def median_filter(predictions, kernel_size=5):
    """
    Applies a median filter to smooth frame-wise class predictions.

    Args:
        predictions (np.ndarray): 1D array of class indices.
        kernel_size (int): Size of the smoothing window. Must be odd.

    Returns:
        np.ndarray: Smoothed 1D array of class indices.
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    if len(predictions) < kernel_size:
        return predictions

    # Apply median filter
    smoothed = scipy.signal.medfilt(predictions, kernel_size=kernel_size)
    return smoothed.astype(int)


def decode_predictions(frame_predictions, min_len=5):
    """
    Decodes frame-wise class indices into an ordered list of gesture labels.
    Applies smoothing, Run-Length Encoding (RLE), and filtering.

    Args:
        frame_predictions (np.ndarray): 1D array of frame-wise class indices.
        min_len (int): Minimum duration (in frames) to keep a gesture segment.

    Returns:
        list[int]: Ordered list of recognized gesture IDs (excluding background).
    """
    # 1. Apply Median Filter to smooth noise
    smoothed_preds = median_filter(frame_predictions, kernel_size=5)

    if len(smoothed_preds) == 0:
        return []

    # 2. Run-Length Encoding (RLE)
    segments = []
    current_label = smoothed_preds[0]
    current_len = 1

    for label in smoothed_preds[1:]:
        if label == current_label:
            current_len += 1
        else:
            segments.append((current_label, current_len))
            current_label = label
            current_len = 1
    segments.append((current_label, current_len))

    # 3. Filter segments
    gesture_list = []
    for label, length in segments:
        # Skip background class
        if label == Config.BACKGROUND_CLASS_ID:
            continue

        # Skip segments that are too short
        if length < min_len:
            continue

        gesture_list.append(int(label))

    return gesture_list


def save_submission(sample_ids, predictions, output_path):
    """
    Saves the final predictions to a CSV file in the required format.
    Format: SessionID,Label1,Label2,Label3

    Args:
        sample_ids (list[str]): List of sample identifiers (e.g., 'Sample00001').
        predictions (list[list[int]]): List of predicted gesture sequences.
        output_path (str): Destination path for the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for sid, preds in zip(sample_ids, predictions):
        # Convert list of ints to comma-separated string
        if preds:
            pred_str = ",".join(map(str, preds))
            row_str = f"{sid},{pred_str}"
        else:
            # Handle empty predictions (no gestures found)
            row_str = f"{sid},"

        rows.append(row_str)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")
