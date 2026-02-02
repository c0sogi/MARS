import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import nltk
from library.config import Config

# ==========================================
# Loss Functions
# ==========================================


class LogSpaceSmoothingLoss(nn.Module):
    """
    Truncated Mean Squared Error loss applied to log-probabilities of adjacent frames.
    Enforces temporal smoothness in predictions.
    """

    def __init__(self, threshold=1.0):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, log_probs):
        """
        Args:
            log_probs: Tensor of shape (Batch, Time, Classes).
                       Should be output of LogSoftmax.
        Returns:
            Scalar loss value.
        """
        # Calculate difference between adjacent frames: t and t+1
        # Shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        se = diff**2

        # Truncate the error to prevent exploding gradients on necessary transitions
        truncated_se = torch.clamp(se, max=self.threshold)

        # Return mean over all dimensions
        return truncated_se.mean()


# ==========================================
# Metrics
# ==========================================


def calculate_levenshtein_distance(preds, targets):
    """
    Computes the Levenshtein distance between two lists of items.

    Args:
        preds (list): List of predicted gesture IDs.
        targets (list): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(preds, targets)


def compute_dataset_metrics(all_preds, all_targets):
    """
    Computes the global metric for the challenge:
    Sum of Levenshtein distances / Total number of ground truth gestures.

    Args:
        all_preds (list of lists): Predicted sequences for the dataset.
        all_targets (list of lists): Ground truth sequences for the dataset.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for p, t in zip(all_preds, all_targets):
        dist = calculate_levenshtein_distance(p, t)
        total_distance += dist
        total_gestures += len(t)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


# ==========================================
# Post-Processing & Decoding
# ==========================================


def run_length_encoding(frame_predictions):
    """
    Converts frame-wise predictions into a list of segments.

    Args:
        frame_predictions (np.array): Array of class IDs of shape (T,).

    Returns:
        list of dict: [{'label': int, 'start': int, 'end': int}, ...]
    """
    if len(frame_predictions) == 0:
        return []

    segments = []
    current_label = frame_predictions[0]
    start_frame = 0

    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != current_label:
            segments.append(
                {"label": int(current_label), "start": start_frame, "end": i - 1}
            )
            current_label = frame_predictions[i]
            start_frame = i

    # Append the last segment
    segments.append(
        {
            "label": int(current_label),
            "start": start_frame,
            "end": len(frame_predictions) - 1,
        }
    )

    return segments


def filter_segments(segments, min_duration=Config.MIN_GESTURE_DURATION):
    """
    Removes segments shorter than the minimum duration.

    Args:
        segments (list of dict): List of segments from run_length_encoding.
        min_duration (int): Minimum frames required.

    Returns:
        list of dict: Filtered segments.
    """
    filtered = []
    for seg in segments:
        duration = seg["end"] - seg["start"] + 1
        if duration >= min_duration:
            filtered.append(seg)
    return filtered


def segments_to_sequence(segments, background_class=Config.BACKGROUND_CLASS_ID):
    """
    Extracts the sequence of gesture IDs from segments, ignoring background.

    Args:
        segments (list of dict): List of segments.
        background_class (int): The class ID to ignore.

    Returns:
        list: Ordered list of gesture IDs.
    """
    sequence = []
    for seg in segments:
        if seg["label"] != background_class:
            sequence.append(seg["label"])
    return sequence


def decode_predictions(frame_probs):
    """
    Full decoding pipeline: Probabilities -> Argmax -> RLE -> Filter -> Sequence.

    Args:
        frame_probs (np.array): Array of shape (T, NumClasses) or (T,).
                                If (T, NumClasses), argmax is applied.

    Returns:
        list: Final predicted sequence of gesture IDs.
    """
    if frame_probs.ndim == 2:
        frame_preds = np.argmax(frame_probs, axis=1)
    else:
        frame_preds = frame_probs

    segments = run_length_encoding(frame_preds)
    valid_segments = filter_segments(segments, min_duration=Config.MIN_GESTURE_DURATION)
    sequence = segments_to_sequence(
        valid_segments, background_class=Config.BACKGROUND_CLASS_ID
    )

    return sequence


# ==========================================
# Data Loading Utilities
# ==========================================


def safe_load_mat(path):
    """
    Safely loads a .mat file, handling potential exceptions.
    """
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def parse_skeleton_structure(video_struct):
    """
    Polymorphic parser for the Skeleton field in the .mat file.
    Handles differences in how Matlab structs are exported (struct array vs cell vs object).

    Args:
        video_struct: The 'Video' object from the loaded .mat file.

    Returns:
        list: List of frame objects/structs containing Skeleton data, or None if failed.
    """
    if not hasattr(video_struct, "Frames"):
        return None

    frames = video_struct.Frames

    # Case 1: Frames is a list or numpy array of objects
    if isinstance(frames, (list, np.ndarray)):
        return frames

    # Case 2: Single object (rare for video, but possible for single frame)
    if hasattr(frames, "Skeleton"):
        return [frames]

    return None
