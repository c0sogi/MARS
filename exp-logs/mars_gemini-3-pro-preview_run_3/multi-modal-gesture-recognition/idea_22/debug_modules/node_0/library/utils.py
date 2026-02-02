import os
import random
import numpy as np
import torch
import scipy.io
import nltk
from library.config import Config


def set_seeds(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_mat_safe(file_path):
    """
    Robustly loads skeleton data from a .mat file, handling polymorphic structures.

    This function implements the 'Polymorphic Parser' logic to address the
    inconsistent data types found in the dataset (e.g., Skeleton fields appearing
    as struct arrays, single objects, or cell arrays).

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: A numpy array of shape (NumFrames, NumJoints, 3) containing
                    WorldPosition coordinates. Returns None if loading fails or
                    structure is invalid.
    """
    try:
        # Load mat file with squeeze_me=True to simplify structure access
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        # print(f"Error loading {file_path}: {e}")
        return None

    if "Video" not in mat:
        return None

    video = mat["Video"]

    # Check for Frames
    if not hasattr(video, "Frames"):
        return None

    frames = video.Frames

    # Handle case where Frames might be a single object instead of an array
    if not isinstance(frames, (np.ndarray, list)):
        frames = [frames]

    num_frames = len(frames)
    num_joints = Config.NUM_JOINTS

    # Initialize skeleton array (T, J, 3)
    # We use 0.0 for missing data, which is standard for padding
    skeleton_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        # Check if Skeleton field exists
        if not hasattr(frame, "Skeleton"):
            continue

        skel_raw = frame.Skeleton

        # --- Polymorphic Parsing Logic ---
        # Case 1: skel_raw is empty, None, or scalar 0
        if skel_raw is None or (np.isscalar(skel_raw) and skel_raw == 0):
            continue

        # Case 2: skel_raw is an array of skeletons (multiple users)
        # We assume the first tracked user is the target
        target_skel = None
        if isinstance(skel_raw, np.ndarray):
            if len(skel_raw) > 0:
                target_skel = skel_raw[0]  # Pick first
        # Case 3: skel_raw is a single struct object
        elif hasattr(skel_raw, "WorldPosition"):
            target_skel = skel_raw
        else:
            # Unknown structure
            continue

        if target_skel is None:
            continue

        # Extract WorldPosition
        if hasattr(target_skel, "WorldPosition"):
            wp = target_skel.WorldPosition

            # WorldPosition should be a struct or object with X, Y, Z
            # Or sometimes it's an array if squeeze_me didn't flatten it perfectly
            try:
                # If it's an object with X, Y, Z attributes (standard case)
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    # X, Y, Z can be arrays of 20 joints
                    # Ensure they are cast to numpy arrays
                    x = np.atleast_1d(wp.X)
                    y = np.atleast_1d(wp.Y)
                    z = np.atleast_1d(wp.Z)

                    if (
                        len(x) == num_joints
                        and len(y) == num_joints
                        and len(z) == num_joints
                    ):
                        skeleton_data[i, :, 0] = x
                        skeleton_data[i, :, 1] = y
                        skeleton_data[i, :, 2] = z

                # Fallback: Check if WorldPosition is a direct 20x3 or 3x20 matrix
                elif isinstance(wp, np.ndarray):
                    if wp.shape == (num_joints, 3):
                        skeleton_data[i] = wp
                    elif wp.shape == (3, num_joints):
                        skeleton_data[i] = wp.T
            except Exception:
                # If extraction fails for this frame, leave as zeros
                continue

    return skeleton_data


def compute_levenshtein_ratio(predictions, ground_truths):
    """
    Computes the Levenshtein distance ratio (Error Rate) for the dataset.

    Metric = (Sum of Levenshtein Distances) / (Total Number of True Gestures)

    Args:
        predictions (list of list of int): Predicted sequences of gesture IDs.
        ground_truths (list of list of int): Ground truth sequences of gesture IDs.

    Returns:
        float: The calculated score. Lower is better.
    """
    total_distance = 0
    total_length = 0

    for pred, true in zip(predictions, ground_truths):
        # Calculate Levenshtein distance
        # nltk.edit_distance computes the minimal edit distance
        d = nltk.edit_distance(pred, true)
        total_distance += d
        total_length += len(true)

    if total_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_length


def rle_encode(frame_predictions, background_label=Config.BACKGROUND_LABEL):
    """
    Performs Run-Length Encoding (RLE) on frame-wise predictions to generate
    the sequence of gesture events.

    Logic:
    1. Collapse consecutive duplicate labels.
    2. Remove the background label.

    Args:
        frame_predictions (list or np.ndarray): List of frame-wise class IDs.
        background_label (int): The ID representing 'no gesture'.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # 1. Collapse consecutive duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # 2. Filter out background labels
    final_sequence = [x for x in collapsed if x != background_label]

    return final_sequence


def get_device():
    """
    Returns the appropriate PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
