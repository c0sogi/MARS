import os
import json
import numpy as np
import pandas as pd
import scipy.io
import nltk
from library.config import Config


def robust_load_mat(file_path):
    """
    Robustly loads a .mat file and extracts skeleton data with physical alignment.
    Handles polymorphic structures (struct arrays vs cells) common in MATLAB exports.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (T, 20, 3) in meters.
                    Returns zeros for missing/invalid frames.
    """
    try:
        # Load with squeeze_me=True to simplify arrays, struct_as_record=False to use objects
        mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

        if not hasattr(mat, "Video") or not hasattr(mat["Video"], "Frames"):
            return np.zeros((0, 20, 3), dtype=np.float32)

        video = mat["Video"]
        frames = video.Frames

        # Handle case where Frames is a single object (1 frame) or array
        if not isinstance(frames, (list, np.ndarray)):
            frames = [frames]

        num_frames = len(frames)
        # 20 joints, 3 coordinates (X, Y, Z)
        skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

        for t, frame in enumerate(frames):
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Handle empty skeleton
            if isinstance(skel, (list, np.ndarray)) and len(skel) == 0:
                continue

            # Identify the user joints structure
            user_joints = None

            # Case 1: skel is an array of 20 joints (direct structure)
            if isinstance(skel, (list, np.ndarray)) and len(skel) == 20:
                # Check if elements look like joints (have WorldPosition)
                if hasattr(skel[0], "WorldPosition"):
                    user_joints = skel

            # Case 2: skel is an array of users (e.g., skel[0] is the user)
            if (
                user_joints is None
                and isinstance(skel, (list, np.ndarray))
                and len(skel) > 0
            ):
                # Take the first tracked user
                first_user = skel[0]
                # Check if this user is a struct array of 20 joints
                if isinstance(first_user, (list, np.ndarray)) and len(first_user) == 20:
                    user_joints = first_user
                # Or maybe the user object has a 'Skeleton' or 'Joints' field?
                # Based on dataset, usually skel is the array of joints if squeezed,
                # or skel[0] is the array of joints.
                elif hasattr(first_user, "WorldPosition"):
                    # If the user array itself contains joints directly
                    user_joints = skel

            # Case 3: skel is a single object (1 joint? or 1 user struct?)
            if user_joints is None and hasattr(skel, "WorldPosition"):
                # Unlikely to be a full skeleton if it's just one object, unless it's a special struct
                pass

            # Fallback: If we couldn't identify a 20-element array, try to treat skel as the joints array
            if user_joints is None and isinstance(skel, (list, np.ndarray)):
                user_joints = skel

            # Extract data
            if user_joints is not None:
                num_joints = min(len(user_joints), 20)
                for j in range(num_joints):
                    joint = user_joints[j]
                    if hasattr(joint, "WorldPosition"):
                        wp = joint.WorldPosition
                        # wp might be a struct with X,Y,Z or an array
                        try:
                            if hasattr(wp, "X"):
                                skeleton_data[t, j, 0] = wp.X
                                skeleton_data[t, j, 1] = wp.Y
                                skeleton_data[t, j, 2] = wp.Z
                            elif isinstance(wp, (list, np.ndarray)) and len(wp) >= 3:
                                skeleton_data[t, j, 0] = wp[0]
                                skeleton_data[t, j, 1] = wp[1]
                                skeleton_data[t, j, 2] = wp[2]
                        except:
                            pass

        # Apply Deterministic Physical Scaling (mm -> meters)
        skeleton_data = skeleton_data * Config.SKELETON_SCALE

        return skeleton_data

    except Exception as e:
        # Return empty array on failure to allow pipeline to continue (e.g. padding will happen later)
        return np.zeros((0, 20, 3), dtype=np.float32)


def compute_levenshtein(predicted_seq, target_seq):
    """
    Computes Levenshtein distance between two sequences of class IDs.

    Args:
        predicted_seq (list): List of predicted class IDs.
        target_seq (list): List of ground truth class IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(predicted_seq, target_seq)


def rle_decode(frame_predictions):
    """
    Run-Length Encoding for frame-wise predictions.

    Args:
        frame_predictions (np.ndarray or list): Sequence of class IDs.

    Returns:
        list of tuples: [(class_id, start_frame, length), ...]
    """
    if len(frame_predictions) == 0:
        return []

    segments = []
    current_label = frame_predictions[0]
    current_start = 0

    for i in range(1, len(frame_predictions)):
        label = frame_predictions[i]
        if label != current_label:
            segments.append((current_label, current_start, i - current_start))
            current_label = label
            current_start = i

    # Append last segment
    segments.append(
        (current_label, current_start, len(frame_predictions) - current_start)
    )

    return segments


def filter_short_segments(segments, min_duration=Config.MIN_GESTURE_FRAMES):
    """
    Filters out short segments and background class (0).

    Args:
        segments (list): List of (label, start, length).
        min_duration (int): Minimum frames to keep a non-background segment.

    Returns:
        list: Filtered list of class IDs.
    """
    filtered_labels = []
    for label, start, length in segments:
        # Assuming 0 is background/null class
        if label == 0:
            continue

        if length >= min_duration:
            filtered_labels.append(label)

    return filtered_labels


def process_predictions(frame_probs_or_ids):
    """
    Full pipeline: Frame Preds -> RLE -> Filter -> Final List.
    Handles both probabilities (argmax) and raw IDs.

    Args:
        frame_probs_or_ids (np.ndarray): Shape (T, C) or (T,).

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if isinstance(frame_probs_or_ids, np.ndarray) and frame_probs_or_ids.ndim > 1:
        # It's probabilities, take argmax
        frame_ids = np.argmax(frame_probs_or_ids, axis=-1)
    else:
        frame_ids = frame_probs_or_ids

    segments = rle_decode(frame_ids)
    final_sequence = filter_short_segments(segments)

    return final_sequence


def load_metadata(split="train"):
    """
    Loads metadata for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Metadata with parsed labels.
    """
    csv_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse labels if present
    if "labels" in df.columns:
        df["labels"] = df["labels"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    return df
