import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def levenshtein_distance(hypothesis, reference):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        hypothesis (list[int]): The predicted sequence of gesture IDs.
        reference (list[int]): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance.
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # Initialize DP matrix
    # Rows: 0 to len_hyp, Cols: 0 to len_ref
    dp = np.zeros((len_hyp + 1, len_ref + 1), dtype=int)

    # Base cases
    for i in range(len_hyp + 1):
        dp[i, 0] = i
    for j in range(len_ref + 1):
        dp[0, j] = j

    # Fill DP table
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i, j] = min(
                dp[i - 1, j] + 1,  # Deletion
                dp[i, j - 1] + 1,  # Insertion
                dp[i - 1, j - 1] + cost,  # Substitution
            )

    return dp[len_hyp, len_ref]


def rle_decode(frame_predictions):
    """
    Converts frame-wise predictions into Run-Length Encoded segments.

    Args:
        frame_predictions (np.ndarray or list): Array of class IDs per frame.

    Returns:
        list[dict]: A list of segments, where each segment is a dict
                    {'label': int, 'start': int, 'end': int}.
    """
    if len(frame_predictions) == 0:
        return []

    segments = []
    curr_label = frame_predictions[0]
    start_idx = 0

    for i in range(1, len(frame_predictions)):
        label = frame_predictions[i]
        if label != curr_label:
            segments.append(
                {"label": int(curr_label), "start": start_idx, "end": i - 1}
            )
            curr_label = label
            start_idx = i

    # Append the last segment
    segments.append(
        {
            "label": int(curr_label),
            "start": start_idx,
            "end": len(frame_predictions) - 1,
        }
    )

    return segments


def process_frame_predictions(frame_preds, min_frames=Config.MIN_GESTURE_FRAMES):
    """
    Post-processes frame-wise predictions to generate the final list of gesture IDs.

    Steps:
    1. Run-Length Encoding to identify segments.
    2. Filter out background class (ID 0).
    3. Filter out segments shorter than min_frames.

    Args:
        frame_preds (np.ndarray or list): Frame-wise class predictions.
        min_frames (int): Minimum duration in frames to keep a gesture.

    Returns:
        list[int]: The ordered list of recognized gesture IDs.
    """
    # 1. Decode into segments
    segments = rle_decode(frame_preds)

    final_sequence = []

    for seg in segments:
        label = seg["label"]
        duration = seg["end"] - seg["start"] + 1

        # 2. Filter Background (0 is background)
        if label == 0:
            continue

        # 3. Filter Short Segments
        if duration < min_frames:
            continue

        final_sequence.append(label)

    return final_sequence
