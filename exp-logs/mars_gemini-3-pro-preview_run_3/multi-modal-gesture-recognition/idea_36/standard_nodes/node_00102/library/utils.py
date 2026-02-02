import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        hyp (list[int]): Hypothesis sequence (predicted labels).
        ref (list[int]): Reference sequence (ground truth labels).

    Returns:
        int: The edit distance.
    """
    n = len(hyp)
    m = len(ref)

    # Optimization: ensure m <= n to minimize space
    if n < m:
        return levenshtein_distance(ref, hyp)

    # Current and previous row for DP
    previous_row = range(m + 1)

    for i, c1 in enumerate(hyp):
        current_row = [i + 1]
        for j, c2 in enumerate(ref):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def run_length_encoding(predictions):
    """
    Compresses frame-wise predictions into segments using Run-Length Encoding.

    Args:
        predictions (list or np.ndarray): Sequence of frame-wise class IDs.

    Returns:
        list[tuple]: A list of segments, where each segment is (class_id, start_frame, end_frame).
    """
    if len(predictions) == 0:
        return []

    segments = []
    current_class = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_class:
            segments.append((current_class, start_frame, i - 1))
            current_class = predictions[i]
            start_frame = i

    # Append the final segment
    segments.append((current_class, start_frame, len(predictions) - 1))

    return segments


def filter_short_segments(segments, min_duration=config.MIN_GESTURE_FRAMES):
    """
    Removes segments that are shorter than the minimum duration threshold.

    Args:
        segments (list[tuple]): List of (class_id, start, end) tuples.
        min_duration (int): Minimum number of frames required to keep a segment.

    Returns:
        list[tuple]: Filtered list of segments.
    """
    filtered = []
    for seg in segments:
        cls_id, start, end = seg
        duration = end - start + 1
        if duration >= min_duration:
            filtered.append(seg)
    return filtered


def decode_predictions(frame_predictions):
    """
    Full decoding pipeline: RLE -> Filter Short Segments -> Remove Background.

    Args:
        frame_predictions (list or np.ndarray): Frame-wise class probabilities or IDs.
                                                If probabilities, argmax is applied.

    Returns:
        list[int]: Ordered list of recognized gesture IDs (excluding background).
    """
    # Handle probability input (if passed as [T, C])
    if isinstance(frame_predictions, np.ndarray) and frame_predictions.ndim > 1:
        frame_predictions = np.argmax(frame_predictions, axis=1)
    elif isinstance(frame_predictions, torch.Tensor):
        if frame_predictions.ndim > 1:
            frame_predictions = torch.argmax(frame_predictions, dim=1)
        frame_predictions = frame_predictions.cpu().numpy()

    # 1. Run-Length Encoding
    segments = run_length_encoding(frame_predictions)

    # 2. Filter Short Segments (Physical Consistency)
    segments = filter_short_segments(segments)

    # 3. Remove Background Class and Extract IDs
    gesture_ids = [
        cls_id for cls_id, _, _ in segments if cls_id != config.BACKGROUND_CLASS_ID
    ]

    return gesture_ids
