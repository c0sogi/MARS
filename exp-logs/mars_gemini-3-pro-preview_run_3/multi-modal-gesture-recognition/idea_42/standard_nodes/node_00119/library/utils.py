import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import pandas as pd
from itertools import groupby
from library.config import Config


def load_mat_robust(file_path):
    """
    Robustly loads skeleton data from a .mat file, handling polymorphic structures.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (NumFrames, NumJoints, 3).
                    Returns None if loading fails or structure is invalid.
    """
    try:
        # Load mat file without squeezing too aggressively initially to preserve structure
        mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

        if "Video" not in mat.__dict__:
            return None

        video = mat.Video

        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames
        num_frames = len(frames)

        # Pre-allocate array: (Time, Joints, 3)
        # Assuming 20 joints based on Config
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            # Check if Skeleton field exists
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Handle Polymorphism: Skeleton might be empty, a struct, or array
            # Case 1: Empty or None
            if skel is None or (isinstance(skel, np.ndarray) and skel.size == 0):
                continue

            # Case 2: Single Skeleton object (Standard case)
            # We need to access WorldPosition

            # Helper to extract WorldPosition from a skeleton object
            def extract_pos(skel_obj):
                if not hasattr(skel_obj, "WorldPosition"):
                    return None
                wp = skel_obj.WorldPosition
                # WorldPosition should have X, Y, Z
                # It might be an object with .X, .Y, .Z or a struct
                try:
                    # Check if it has X, Y, Z attributes
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        # Assuming X, Y, Z are arrays of size (NumJoints,)
                        # Stack them
                        return np.stack([wp.X, wp.Y, wp.Z], axis=1)
                    # Sometimes it might be a simple array?
                    # Based on dataset description, it's a struct with X,Y,Z
                except Exception:
                    return None
                return None

            # If skel is an array of objects (multiple skeletons?), we usually take the first tracked one
            # But the dataset description implies "a skeleton".
            # If it's an array (e.g. shape (1,)), extract the item.
            if isinstance(skel, np.ndarray):
                if skel.size > 0:
                    # Take the first one if multiple exist (UserIndex usually filters this,
                    # but here we just want raw data)
                    current_skel = (
                        skel if not isinstance(skel, np.ndarray) else skel.flat[0]
                    )
                else:
                    continue
            else:
                current_skel = skel

            pos = extract_pos(current_skel)

            if pos is not None:
                # Ensure shape matches (NumJoints, 3)
                if pos.shape == (Config.NUM_JOINTS, 3):
                    skeleton_data[i] = pos
                else:
                    # Handle mismatch if necessary, or skip
                    pass

        return skeleton_data

    except Exception as e:
        # In production, we might log this. For now, return None to signal failure.
        return None


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.

    Args:
        seq1 (list[int]): Predicted sequence.
        seq2 (list[int]): Ground truth sequence.

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


def decode_predictions(probabilities, min_duration=Config.MIN_GESTURE_DURATION):
    """
    Decodes frame-wise probabilities into a sequence of gesture IDs using
    Run-Length Encoding and duration filtering.

    Args:
        probabilities (np.ndarray or torch.Tensor): Shape (T, NumClasses).
        min_duration (int): Minimum frames to consider a segment valid.

    Returns:
        list[int]: Ordered list of recognized gesture IDs (excluding background).
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    # 1. Argmax to get class indices
    pred_indices = np.argmax(probabilities, axis=1)

    # 2. Run-Length Encoding
    decoded_sequence = []

    # groupby returns consecutive keys and an iterator over the group
    for k, g in groupby(pred_indices):
        length = len(list(g))

        # 3. Filter
        # Must not be background class (0)
        # Must meet minimum duration
        if k != Config.BACKGROUND_CLASS_ID and length >= min_duration:
            decoded_sequence.append(int(k))

    return decoded_sequence


class LogSpaceSmoothingLoss(nn.Module):
    """
    Truncated MSE loss applied to log-probabilities to enforce temporal smoothness.
    L = mean( min( || log(p_t) - log(p_{t-1}) ||^2, threshold ) )
    """

    def __init__(self, threshold=Config.SMOOTHING_THRESHOLD):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs):
        """
        Args:
            log_probs (torch.Tensor): Shape (Batch, Time, Classes).
                                      Should be output of F.log_softmax.
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Calculate difference between adjacent time steps
        # diff: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        squared_diff = diff**2

        # Sum over classes to get distance per time step
        # dist: (Batch, Time-1)
        dist = torch.sum(squared_diff, dim=2)

        # Truncate (Clamp)
        truncated_dist = torch.clamp(dist, max=self.threshold)

        # Mean over batch and time
        loss = torch.mean(truncated_dist)

        return loss


def save_predictions(predictions_dict, output_path=Config.SUBMISSION_FILE):
    """
    Saves predictions to a CSV file in the submission format.

    Args:
        predictions_dict (dict): Map from sequence_id (str) to list of gesture IDs (list[int]).
        output_path (str): Path to save the CSV.
    """
    rows = []
    for seq_id, gestures in predictions_dict.items():
        # Format: Session00001,2,12,3
        # If gestures is empty, it will be "Session00001" (or handled gracefully)
        gesture_str = ",".join(map(str, gestures))
        if gesture_str:
            row_str = f"{seq_id},{gesture_str}"
        else:
            row_str = f"{seq_id}"  # No gestures predicted
        rows.append(row_str)

    # Write to file
    # Note: The submission format example shows no header, just lines.
    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")


def pad_sequence_ndarray(sequences, max_len=None, padding_value=0.0):
    """
    Pads a list of numpy arrays (T, F) to (Batch, MaxT, F).
    """
    if not sequences:
        return np.array([])

    if max_len is None:
        max_len = max(len(s) for s in sequences)

    feature_dim = sequences[0].shape[1]
    batch_size = len(sequences)

    padded = np.full(
        (batch_size, max_len, feature_dim), padding_value, dtype=np.float32
    )

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]

    return padded
