import os
import random
import numpy as np
import torch
from scipy.ndimage import median_filter as scipy_median_filter
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein edit distance between two sequences.
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
                cost = 0
            else:
                cost = 1

            matrix[x, y] = min(
                matrix[x - 1, y] + 1,  # Deletion
                matrix[x, y - 1] + 1,  # Insertion
                matrix[x - 1, y - 1] + cost,  # Substitution
            )
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the competition metric: Sum of Levenshtein distances divided by
    total number of ground truth gestures.

    Args:
        predictions: List of lists containing predicted gesture IDs.
        ground_truths: List of lists containing ground truth gesture IDs.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_truth_gestures = 0

    for p, t in zip(predictions, ground_truths):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_truth_gestures += len(t)

    if total_truth_gestures == 0:
        return 0.0

    return total_distance / total_truth_gestures


def median_filter(predictions, window_size=Config.MEDIAN_FILTER_SIZE):
    """
    Applies a median filter to smooth frame-wise predictions.

    Args:
        predictions: 1D array-like of class indices.
        window_size: Size of the sliding window.

    Returns:
        np.ndarray: Smoothed predictions.
    """
    preds = np.array(predictions)
    # mode='nearest' repeats the edge values to handle boundaries
    smoothed = scipy_median_filter(preds, size=window_size, mode="nearest")
    return smoothed


def rle_decode(predictions, min_length=Config.MIN_GESTURE_LENGTH, background_class=0):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using Run-Length Encoding.
    Filters out background class and segments shorter than min_length.

    Args:
        predictions: 1D array-like of class indices.
        min_length: Minimum number of frames to consider a valid gesture instance.
        background_class: The class ID representing background/no-gesture.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    decoded_gestures = []
    current_val = predictions[0]
    current_len = 1

    for i in range(1, len(predictions)):
        val = predictions[i]
        if val == current_val:
            current_len += 1
        else:
            # End of a run
            if current_val != background_class and current_len >= min_length:
                decoded_gestures.append(int(current_val))

            current_val = val
            current_len = 1

    # Handle the final run
    if current_val != background_class and current_len >= min_length:
        decoded_gestures.append(int(current_val))

    return decoded_gestures


def generate_submission(predictions_dict, output_path=Config.SUBMISSION_PATH):
    """
    Generates the submission CSV file in the format: SessionID,Label1,Label2,...

    Args:
        predictions_dict: Dictionary mapping sample_id (str) to list of gesture IDs (int).
        output_path: Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    # Sort IDs for deterministic output order
    sorted_ids = sorted(predictions_dict.keys())

    for sample_id in sorted_ids:
        gestures = predictions_dict[sample_id]
        sid = sample_id.strip()

        if not gestures:
            # If no gestures detected, just write the ID
            line = f"{sid}"
        else:
            labels_str = ",".join(map(str, gestures))
            line = f"{sid},{labels_str}"

        rows.append(line)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")

    print(f"Submission saved to {output_path}")
