import os
import random
import numpy as np
import torch
import csv
from numpy.lib.stride_tricks import sliding_window_view
from library.config import Config


def set_seed(seed=Config.SEED):
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


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.
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


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the global normalized Levenshtein distance metric.

    Args:
        predictions: List of lists, where each inner list contains predicted gesture IDs.
        ground_truths: List of lists, where each inner list contains true gesture IDs.

    Returns:
        float: The sum of Levenshtein distances divided by the total number of ground truth gestures.
    """
    total_distance = 0
    total_truth_length = 0

    for pred, truth in zip(predictions, ground_truths):
        dist = levenshtein_distance(pred, truth)
        total_distance += dist
        total_truth_length += len(truth)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length


def median_filter_1d(arr, kernel_size=7):
    """
    Applies a 1D median filter to the input array with Nearest-Neighbor padding.

    Args:
        arr: 1D numpy array of class indices.
        kernel_size: Size of the median filter window (must be odd).

    Returns:
        Filtered 1D numpy array.
    """
    arr = np.array(arr)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad_width = kernel_size // 2
    # Nearest-neighbor padding (edge padding)
    padded = np.pad(arr, (pad_width, pad_width), mode="edge")

    # Create sliding windows
    windows = sliding_window_view(padded, window_shape=kernel_size)

    # Compute median for each window
    # Using np.median usually returns float, cast back to appropriate type
    filtered = np.median(windows, axis=1).astype(arr.dtype)

    return filtered


def decode_predictions(frame_preds, background_class=0):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs.
    Collapses consecutive duplicates and removes background class.

    Args:
        frame_preds: List or array of frame-wise class indices.
        background_class: The class ID representing background/no-gesture.

    Returns:
        List of gesture IDs.
    """
    if len(frame_preds) == 0:
        return []

    # Collapse repeats
    collapsed = []
    for i, pred in enumerate(frame_preds):
        if i == 0 or pred != frame_preds[i - 1]:
            collapsed.append(pred)

    # Remove background
    gestures = [g for g in collapsed if g != background_class]

    return gestures


def generate_submission_file(predictions_dict, output_path):
    """
    Generates the submission CSV file in the format: Id,Sequence

    Args:
        predictions_dict: Dictionary mapping sample_id (str) to list of gesture IDs (int).
                          e.g., {'Sample00300': [2, 12, 3]}
        output_path: Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by ID to ensure consistent order
    sorted_ids = sorted(predictions_dict.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Sequence"])

        for sample_id in sorted_ids:
            # Extract numeric ID from 'SampleXXXXX'
            try:
                numeric_id = int(sample_id.replace("Sample", ""))
            except ValueError:
                # Fallback if ID format is unexpected
                numeric_id = sample_id

            preds = predictions_dict[sample_id]
            # Format sequence as space-separated string
            seq_str = " ".join(map(str, preds))

            writer.writerow([numeric_id, seq_str])
