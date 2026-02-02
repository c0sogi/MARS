import numpy as np
import logging
import sys
import os
from library.config import config


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences (hypothesis and reference).
    This is the core metric for the challenge (edit distance).

    Args:
        hyp (list): The predicted sequence of labels.
        ref (list): The ground truth sequence of labels.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    n = len(hyp)
    m = len(ref)

    # Initialize DP matrix
    # dp[i][j] is the distance between hyp[:i] and ref[:j]
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def run_length_encoding(predictions):
    """
    Converts a sequence of frame-wise predictions into a list of segments.

    Args:
        predictions (list or np.ndarray): 1D array of class indices per frame.

    Returns:
        list of dict: A list of segment dictionaries, each containing:
                      {'label': int, 'start': int, 'end': int, 'duration': int}
    """
    if len(predictions) == 0:
        return []

    segments = []
    current_label = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            segments.append(
                {
                    "label": int(current_label),
                    "start": start_frame,
                    "end": i - 1,
                    "duration": i - start_frame,
                }
            )
            current_label = predictions[i]
            start_frame = i

    # Append the final segment
    segments.append(
        {
            "label": int(current_label),
            "start": start_frame,
            "end": len(predictions) - 1,
            "duration": len(predictions) - start_frame,
        }
    )

    return segments


def filter_short_segments(segments, min_duration=None):
    """
    Removes segments that are shorter than the specified minimum duration.
    This enforces the physical constraint that gestures must have a minimum temporal extent.

    Args:
        segments (list of dict): List of segments produced by run_length_encoding.
        min_duration (int, optional): Minimum frame count. Defaults to config.MIN_GESTURE_DURATION.

    Returns:
        list of dict: The filtered list of segments.
    """
    if min_duration is None:
        min_duration = config.MIN_GESTURE_DURATION

    filtered_segments = [seg for seg in segments if seg["duration"] >= min_duration]
    return filtered_segments


def process_predictions_for_submission(frame_preds, background_class=0):
    """
    Processes raw frame-wise predictions into the final list of gesture IDs required for submission.
    Pipeline: RLE -> Duration Filtering -> Background Removal -> ID Extraction.

    Args:
        frame_preds (list or np.ndarray): Raw frame predictions.
        background_class (int): The class ID representing background (default 0).

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # 1. Convert to segments
    segments = run_length_encoding(frame_preds)

    # 2. Filter noise (short segments)
    segments = filter_short_segments(segments)

    # 3. Extract non-background labels
    result_labels = []
    for seg in segments:
        if seg["label"] != background_class:
            result_labels.append(seg["label"])

    return result_labels


def compute_sequence_accuracy(hyp_sequences, ref_sequences):
    """
    Computes the global error rate based on the Levenshtein distance.
    Metric = (Sum of Levenshtein Distances) / (Total Number of Ground Truth Gestures).

    Args:
        hyp_sequences (list of list): List of predicted gesture ID lists.
        ref_sequences (list of list): List of ground truth gesture ID lists.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_ref_gestures = 0

    # Iterate over all samples
    # Note: We assume lists are aligned by index
    for hyp, ref in zip(hyp_sequences, ref_sequences):
        dist = levenshtein_distance(hyp, ref)
        total_distance += dist
        total_ref_gestures += len(ref)

    if total_ref_gestures == 0:
        return 0.0

    return total_distance / total_ref_gestures


def setup_logger(log_file=None):
    """
    Configures a standard logger for the application.

    Args:
        log_file (str, optional): Path to a file where logs should be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("GestureRecognition")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (optional)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
