import numpy as np
import scipy.io
from library.config import Config


def robust_load_mat(file_path):
    """
    Robustly loads skeleton data from a .mat file, handling polymorphic structures
    (struct arrays, single structs, primitive arrays) to prevent parsing failures.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (NumFrames, NumJoints, 3).
                    Returns a zero-filled array if loading fails or data is missing.
    """
    # Initialize with zeros for safety
    default_skel = np.zeros((1, Config.NUM_JOINTS, 3), dtype=np.float32)

    try:
        # Load mat file with squeeze_me=True to simplify structure access
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default_skel

    if not hasattr(mat, "Video"):
        return default_skel

    video = mat["Video"]

    # Handle 0-d array wrapping (common with squeeze_me=True for single structs)
    if isinstance(video, np.ndarray):
        if video.ndim == 0:
            video = video.item()
        elif video.size >= 1:
            # If it's an array of structs, usually the data is in the first element
            try:
                video = video.flat[0]
            except:
                pass

    # Handle Frames structure
    if not hasattr(video, "Frames"):
        return default_skel

    frames = video.Frames

    # Polymorphism: Frames can be a single struct (if 1 frame), 0-d array, or array
    if isinstance(frames, np.ndarray):
        if frames.ndim == 0:
            frames = np.array([frames.item()])
    elif not isinstance(frames, (list, np.ndarray)):
        # If it's a single object (scipy.io.matlab.mat_struct), wrap it
        frames = np.array([frames])

    num_frames = len(frames)
    if num_frames == 0:
        return default_skel

    skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        # Check if Skeleton field exists
        if not hasattr(frame, "Skeleton"):
            continue

        skel = frame.Skeleton

        # Unwrap 0-d array if necessary
        if isinstance(skel, np.ndarray) and skel.ndim == 0:
            skel = skel.item()

        # Polymorphism: Skeleton can be None, NaN, empty array, single struct, or array of structs

        # Case 1: Empty or NaN (Tracking lost)
        if skel is None:
            continue
        if isinstance(skel, float) and np.isnan(skel):
            continue
        if isinstance(skel, np.ndarray) and skel.size == 0:
            continue

        # Case 2: Array of joints (Standard case)
        # We expect 20 joints.
        if isinstance(skel, (list, np.ndarray)):
            # Take up to NUM_JOINTS to avoid index errors
            limit = min(len(skel), Config.NUM_JOINTS)
            for j in range(limit):
                joint = skel[j]
                if hasattr(joint, "WorldPosition"):
                    wp = joint.WorldPosition
                    # Check for X, Y, Z fields
                    if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                        skeleton_data[i, j, 0] = float(wp.X)
                        skeleton_data[i, j, 1] = float(wp.Y)
                        skeleton_data[i, j, 2] = float(wp.Z)

        # Case 3: Single struct (Edge case: only 1 joint tracked or wrapped weirdly)
        elif hasattr(skel, "WorldPosition"):
            # Try to extract position for the first joint index
            wp = skel.WorldPosition
            if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                skeleton_data[i, 0, 0] = float(wp.X)
                skeleton_data[i, 0, 1] = float(wp.Y)
                skeleton_data[i, 0, 2] = float(wp.Z)

    return skeleton_data


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences of integers
    using the standard Dynamic Programming algorithm.
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


def run_length_encoding(predictions):
    """
    Decodes frame-wise predictions into a list of gesture IDs.
    Collapses consecutive duplicates and removes background class (0).

    Args:
        predictions (list or np.array): Frame-wise class IDs.

    Returns:
        list: Sequence of gesture IDs.
    """
    if len(predictions) == 0:
        return []

    decoded = []
    prev = -1

    for label in predictions:
        label = int(label)
        # Skip background class (0)
        if label == 0:
            prev = 0
            continue

        # If new label is different from previous, add it
        # Note: If we had [1, 0, 1], this produces [1, 1], which is correct
        # as the background separates two instances of the same gesture.
        if label != prev:
            decoded.append(label)
            prev = label

    return decoded


def calculate_score(predictions_dict, ground_truth_dict):
    """
    Calculates the competition metric:
    Sum of Levenshtein distances / Total number of gestures in ground truth.

    Args:
        predictions_dict (dict): {sample_id: [list of gesture IDs]}
        ground_truth_dict (dict): {sample_id: [list of gesture IDs]}

    Returns:
        float: The error rate score.
    """
    total_distance = 0
    total_gestures = 0

    for sample_id, gt_seq in ground_truth_dict.items():
        pred_seq = predictions_dict.get(sample_id, [])

        dist = compute_levenshtein(pred_seq, gt_seq)
        total_distance += dist
        total_gestures += len(gt_seq)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
