import numpy as np
import scipy.io
from library.config import Config


def load_mat_file_polymorphic(mat_path):
    """
    Robustly loads a .mat file and extracts skeleton data, handling various
    internal structures (struct arrays, cells, etc.) found in the dataset.

    Args:
        mat_path (str): Path to the .mat file.

    Returns:
        np.ndarray: Skeleton data of shape (T, NumJoints, 3).
                    Returns None if loading fails or structure is invalid.
    """
    try:
        # Load with squeeze_me=True to simplify arrays, struct_as_record=False to use objects
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat.__dict__:
            return None

        video = mat.Video

        # Check for Frames
        if not hasattr(video, "Frames"):
            return None

        frames = video.Frames

        # Handle case where Frames is a single object (1 frame) vs array
        if not isinstance(frames, (np.ndarray, list)):
            frames = [frames]

        num_frames = len(frames)
        if num_frames == 0:
            return None

        # Initialize container: (T, 20, 3)
        # We assume 20 joints as per dataset spec
        skeleton_data = np.zeros((num_frames, Config.NUM_JOINTS, 3), dtype=np.float32)

        for i, frame in enumerate(frames):
            # Check if Skeleton exists
            if not hasattr(frame, "Skeleton"):
                continue

            skel = frame.Skeleton

            # Handle potential array of skeletons (multiple users) -> take first
            target_skel = None
            if isinstance(skel, (np.ndarray, list)):
                if len(skel) > 0:
                    target_skel = skel[0]
            else:
                target_skel = skel

            if target_skel is None or not hasattr(target_skel, "WorldPosition"):
                continue

            world_pos = target_skel.WorldPosition

            # Extract coordinates based on structure type
            try:
                # Case 1: WorldPosition has X, Y, Z attributes (common in this dataset)
                if (
                    hasattr(world_pos, "X")
                    and hasattr(world_pos, "Y")
                    and hasattr(world_pos, "Z")
                ):
                    x = np.atleast_1d(world_pos.X)
                    y = np.atleast_1d(world_pos.Y)
                    z = np.atleast_1d(world_pos.Z)

                    if len(x) == Config.NUM_JOINTS:
                        skeleton_data[i, :, 0] = x
                        skeleton_data[i, :, 1] = y
                        skeleton_data[i, :, 2] = z

                # Case 2: WorldPosition is a raw matrix
                elif isinstance(world_pos, (np.ndarray, list)):
                    wp_arr = np.array(world_pos)
                    if wp_arr.shape == (Config.NUM_JOINTS, 3):
                        skeleton_data[i] = wp_arr
                    elif wp_arr.shape == (3, Config.NUM_JOINTS):
                        skeleton_data[i] = wp_arr.T
            except Exception:
                # If parsing fails for a specific frame, leave as zeros
                pass

        return skeleton_data

    except Exception:
        return None


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of labels.

    Args:
        seq1 (list or np.array): First sequence of class IDs.
        seq2 (list or np.array): Second sequence of class IDs.

    Returns:
        int: The Levenshtein distance.
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
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )

    return matrix[size_x - 1, size_y - 1]


def run_length_encoding(predictions):
    """
    Converts frame-wise predictions into a list of segments.

    Args:
        predictions (np.array): Array of frame-wise class IDs.

    Returns:
        list of dict: [{'label': int, 'start': int, 'end': int}, ...]
    """
    if len(predictions) == 0:
        return []

    segments = []
    current_label = predictions[0]
    start_idx = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            segments.append(
                {"label": int(current_label), "start": start_idx, "end": i - 1}
            )
            current_label = predictions[i]
            start_idx = i

    # Add the last segment
    segments.append(
        {"label": int(current_label), "start": start_idx, "end": len(predictions) - 1}
    )

    return segments


def filter_short_segments(segments, min_duration=Config.MIN_DURATION):
    """
    Removes segments that are shorter than the minimum duration.

    Args:
        segments (list of dict): List of segments from run_length_encoding.
        min_duration (int): Minimum number of frames required.

    Returns:
        list of dict: Filtered segments.
    """
    filtered = []
    for seg in segments:
        duration = seg["end"] - seg["start"] + 1
        if duration >= min_duration:
            filtered.append(seg)
    return filtered


def decode_predictions_to_labels(frame_probs):
    """
    Full decoding pipeline: Argmax -> RLE -> Filter -> Remove Background.

    Args:
        frame_probs (np.ndarray): Probability matrix (T, NumClasses).

    Returns:
        list: Ordered list of recognized gesture IDs (excluding background).
    """
    # 1. Argmax
    pred_labels = np.argmax(frame_probs, axis=1)

    # 2. RLE
    segments = run_length_encoding(pred_labels)

    # 3. Filter Short Segments
    valid_segments = filter_short_segments(segments, min_duration=Config.MIN_DURATION)

    # 4. Extract Labels (removing background class 0)
    final_labels = []
    for seg in valid_segments:
        label = seg["label"]
        if label != 0:  # Assuming 0 is background
            final_labels.append(label)

    return final_labels


def calculate_score(ground_truth_dict, predictions_dict):
    """
    Calculates the competition metric: Mean Levenshtein Distance.

    Args:
        ground_truth_dict (dict): {sample_id: [label1, label2, ...]}
        predictions_dict (dict): {sample_id: [label1, label2, ...]}

    Returns:
        float: The score (Error Rate).
    """
    total_dist = 0
    total_gestures = 0

    for sample_id, true_labels in ground_truth_dict.items():
        pred_labels = predictions_dict.get(sample_id, [])

        dist = compute_levenshtein(true_labels, pred_labels)
        total_dist += dist
        total_gestures += len(true_labels)

    if total_gestures == 0:
        return 0.0

    return total_dist / total_gestures
