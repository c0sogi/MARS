import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from itertools import groupby
from library.config import TRUNCATION_THRESHOLD, MIN_GESTURE_DURATION, NUM_JOINTS

# ==========================================
# Custom Loss Classes
# ==========================================


class LogSpaceSmoothingLoss(nn.Module):
    """
    Truncated MSE loss applied to the log-probabilities of adjacent frames.
    Encourages temporal smoothness in predictions while allowing for sharp transitions
    by truncating the penalty for large differences.
    """

    def __init__(self, threshold=TRUNCATION_THRESHOLD):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, log_probs):
        """
        Args:
            log_probs: Tensor of shape (Batch, Time, Classes).
                       It is expected to be log-probabilities (output of LogSoftmax).
        Returns:
            Scalar loss value.
        """
        # Calculate difference between adjacent time steps: t and t-1
        # diff shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Square the differences
        diff_sq = diff**2

        # Truncate the squared difference to avoid penalizing valid sharp transitions too heavily
        truncated = torch.clamp(diff_sq, max=self.threshold_sq)

        # Return the mean loss
        return truncated.mean()


# ==========================================
# Metrics & Evaluation
# ==========================================


def levenshtein_distance(p, y):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        p (list): Predicted sequence of labels.
        y (list): Ground truth sequence of labels.

    Returns:
        int: The edit distance.
    """
    m = len(p)
    n = len(y)

    # Initialize DP matrix
    # D[i, j] is distance between first i elements of p and first j elements of y
    D = np.zeros((m + 1, n + 1), dtype=int)

    # Initialization
    for i in range(m + 1):
        D[i, 0] = i
    for j in range(n + 1):
        D[0, j] = j

    # Recurrence
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == y[j - 1]:
                cost = 0
            else:
                cost = 1

            D[i, j] = min(
                D[i - 1, j] + 1,  # Deletion
                D[i, j - 1] + 1,  # Insertion
                D[i - 1, j - 1] + cost,  # Substitution
            )

    return D[m, n]


def compute_levenshtein_ratio(predictions, ground_truths):
    """
    Calculates the competition metric: Sum of Levenshtein distances divided by
    total number of ground truth gestures.

    Args:
        predictions (list of lists): Predicted gesture IDs for each sequence.
        ground_truths (list of lists): Ground truth gesture IDs for each sequence.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for p, gt in zip(predictions, ground_truths):
        dist = levenshtein_distance(p, gt)
        total_distance += dist
        total_gestures += len(gt)

    if total_gestures == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_gestures


# ==========================================
# Post-Processing
# ==========================================


def filter_short_segments(
    frame_predictions, min_duration=MIN_GESTURE_DURATION, background_class=0
):
    """
    Applies Run-Length Encoding (RLE) to frame-wise predictions and filters out
    segments shorter than min_duration. Returns the final list of gesture IDs.

    Args:
        frame_predictions (list or np.array): Sequence of frame labels.
        min_duration (int): Minimum number of frames for a gesture to be considered valid.
        background_class (int): The label ID for the background/null class.

    Returns:
        list: Ordered list of recognized gesture IDs (excluding background).
    """
    if isinstance(frame_predictions, torch.Tensor):
        frame_predictions = frame_predictions.cpu().numpy()

    filtered_gestures = []

    # Group consecutive identical labels
    for label, group in groupby(frame_predictions):
        # Calculate length of the segment
        length = sum(1 for _ in group)

        # Filter logic: Must be non-background and sufficiently long
        if label != background_class and length >= min_duration:
            filtered_gestures.append(int(label))

    return filtered_gestures


# ==========================================
# Data Processing Utilities
# ==========================================


def load_skeleton_data(mat_path):
    """
    Robustly parses the .mat file to extract skeleton joint positions.
    Handles polymorphic structures (struct array vs cell array) found in the dataset.

    Args:
        mat_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (NumFrames, NumJoints, 3).
                    Returns None if parsing fails or data is invalid.
    """
    try:
        # Load mat file, struct_as_record=False allows dot notation access
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if not hasattr(mat, "Video") or not hasattr(mat.Video, "Frames"):
            return None

        frames = mat.Video.Frames
        num_frames = len(frames) if isinstance(frames, (list, np.ndarray)) else 0

        if num_frames == 0:
            return None

        # Pre-allocate: (Time, Joints, 3)
        skeleton_data = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float32)

        # Iterate through frames to extract WorldPosition
        for i, frame in enumerate(frames):
            # Check if Skeleton exists and has WorldPosition
            if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
                wp = frame.Skeleton.WorldPosition

                # WorldPosition can be a struct with X,Y,Z fields or an array
                # Based on dataset description, it's likely a struct or 20x3 array.
                # We need to handle the specific format of this dataset.
                # Assuming standard Kinect format where WorldPosition is 20x3 or struct array.

                # Case 1: WorldPosition is an array (20, 3) or similar
                if isinstance(wp, np.ndarray) and wp.shape == (NUM_JOINTS, 3):
                    skeleton_data[i] = wp

                # Case 2: WorldPosition is a struct with X, Y, Z (scalar or array)
                # This part depends heavily on the specific mat file structure variation.
                # We implement a robust fallback extraction.
                elif hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # If X, Y, Z are arrays of length 20
                    try:
                        x = np.atleast_1d(wp.X)
                        y = np.atleast_1d(wp.Y)
                        z = np.atleast_1d(wp.Z)
                        if len(x) == NUM_JOINTS:
                            skeleton_data[i, :, 0] = x
                            skeleton_data[i, :, 1] = y
                            skeleton_data[i, :, 2] = z
                    except:
                        pass  # Keep zero if extraction fails
            else:
                # If skeleton data is missing for a frame, we keep it as zeros
                # (or could interpolate later)
                pass

        return skeleton_data

    except Exception as e:
        # print(f"Warning: Failed to load skeleton for {mat_path}: {e}")
        return None


def compute_kinematics(skeleton_data):
    """
    Computes Velocity and Acceleration from raw skeleton positions.

    Args:
        skeleton_data (np.ndarray): Shape (T, J, 3).

    Returns:
        np.ndarray: Concatenated features (Position, Velocity, Acceleration)
                    of shape (T, J, 9).
    """
    # skeleton_data: (T, J, 3)

    # Pad to maintain temporal dimension
    # Velocity: P_t - P_{t-1}
    # We pad the first frame with 0 velocity
    velocity = np.diff(skeleton_data, axis=0, prepend=skeleton_data[0:1])

    # Acceleration: V_t - V_{t-1}
    # We pad the first frame with 0 acceleration
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # Concatenate: (T, J, 3+3+3) = (T, J, 9)
    features = np.concatenate([skeleton_data, velocity, acceleration], axis=2)

    return features
