import os
import random
import numpy as np
import torch
import scipy.io
from library.config import Config


def set_seeds(seed=Config.SEED):
    """
    Sets random seeds for reproducibility across Python, Numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_load_mat(path):
    """
    Robustly loads a .mat file, handling different internal structures (structs vs arrays).
    Implements the 'Polymorphic Parser' requirement to prevent silent failures.

    Args:
        path (str): Path to the .mat file.

    Returns:
        object: The loaded matlab object/struct, or None if loading fails.
    """
    try:
        # loadmat with struct_as_record=False creates objects for structs
        # squeeze_me=True removes length-1 dimensions, simplifying access
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences of integers using Dynamic Programming.

    Args:
        seq1 (list[int]): First sequence.
        seq2 (list[int]): Second sequence.

    Returns:
        float: The edit distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

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


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the competition metric: Sum(Levenshtein) / Total GT Gestures.

    Args:
        predictions (dict): {sample_id: [label_id, ...]}
        ground_truths (dict): {sample_id: [label_id, ...]}

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    # Iterate over ground truths to ensure we cover all validation samples
    for sample_id, gt_seq in ground_truths.items():
        pred_seq = predictions.get(sample_id, [])

        # Calculate distance
        dist = levenshtein_distance(pred_seq, gt_seq)
        total_distance += dist
        total_gestures += len(gt_seq)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def run_length_encoding(frame_predictions):
    """
    Converts a sequence of frame-wise class IDs into temporal segments.

    Args:
        frame_predictions (list or np.array): Sequence of class IDs.

    Returns:
        list of tuples: (class_id, start_frame, end_frame)
    """
    if len(frame_predictions) == 0:
        return []

    segments = []
    current_class = frame_predictions[0]
    start_frame = 0

    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != current_class:
            segments.append((current_class, start_frame, i - 1))
            current_class = frame_predictions[i]
            start_frame = i

    # Add the last segment
    segments.append((current_class, start_frame, len(frame_predictions) - 1))

    return segments


def filter_segments(
    segments,
    min_duration=Config.MIN_GESTURE_DURATION,
    background_class=Config.BACKGROUND_CLASS_ID,
):
    """
    Filters segments based on duration and removes background class.

    Args:
        segments (list): List of (class_id, start, end) tuples.
        min_duration (int): Minimum length in frames to retain a segment.
        background_class (int): Class ID to remove (typically 0).

    Returns:
        list[int]: The ordered list of retained gesture IDs.
    """
    final_sequence = []

    for class_id, start, end in segments:
        duration = end - start + 1

        # Skip background
        if class_id == background_class:
            continue

        # Filter short segments
        if duration < min_duration:
            continue

        final_sequence.append(int(class_id))

    return final_sequence


def decode_predictions_to_sequence(frame_probs):
    """
    Decodes frame probabilities into the final list of gesture IDs.
    Applies Argmax -> RLE -> Duration Filtering -> Background Removal.

    Args:
        frame_probs (np.array): Array of shape (T, NumClasses).

    Returns:
        list[int]: The predicted sequence of gesture IDs.
    """
    # 1. Argmax to get class indices
    frame_preds = np.argmax(frame_probs, axis=1)

    # 2. Run-Length Encoding
    segments = run_length_encoding(frame_preds)

    # 3. Filter short segments and remove background
    sequence = filter_segments(segments)

    return sequence


def save_submission(predictions, output_path=Config.SUBMISSION_PATH):
    """
    Saves predictions to a CSV file in the submission format.

    Args:
        predictions (dict): {sample_id: [label_id, ...]}
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Id,Sequence\n")
        for sample_id, seq in predictions.items():
            # Sanitize ID (Cite debug_lesson_5)
            clean_id = sample_id
            if isinstance(clean_id, str):
                digits = "".join(filter(str.isdigit, clean_id))
                if digits:
                    clean_id = str(int(digits))

            # Format: Id,Sequence (space separated) (Cite debug_lesson_1)
            seq_str = " ".join([str(x) for x in seq])
            f.write(f"{clean_id},{seq_str}\n")
