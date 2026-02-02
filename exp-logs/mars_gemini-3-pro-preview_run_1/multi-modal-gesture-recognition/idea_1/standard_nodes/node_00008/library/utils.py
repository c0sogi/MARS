import numpy as np
import pandas as pd
import os
from library.config import SUBMISSION_DIR


def calculate_levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.

    Args:
        seq1 (list): First sequence of items (e.g., gesture IDs).
        seq2 (list): Second sequence of items.

    Returns:
        int: The edit distance.
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
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return int(matrix[size_x - 1, size_y - 1])


def compute_challenge_score(ground_truths, predictions):
    """
    Computes the challenge metric: Total Levenshtein Distance / Total True Gestures.

    Args:
        ground_truths (list of lists): List of ground truth sequences.
        predictions (list of lists): List of predicted sequences.

    Returns:
        float: The normalized error score.
    """
    total_distance = 0
    total_true_gestures = 0

    for gt, pred in zip(ground_truths, predictions):
        dist = calculate_levenshtein_distance(gt, pred)
        total_distance += dist
        total_true_gestures += len(gt)

    if total_true_gestures == 0:
        return 0.0

    return total_distance / total_true_gestures


def median_filter_1d(data, kernel_size=5):
    """
    Applies a 1D median filter to the input array.

    Args:
        data (np.array): Input 1D array.
        kernel_size (int): Size of the window. Must be odd.

    Returns:
        np.array: Filtered array.
    """
    if kernel_size <= 1:
        return data

    pad_size = kernel_size // 2
    # Pad with edge values to maintain length
    padded = np.pad(data, (pad_size, pad_size), mode="edge")
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    # Compute median along the window axis
    return np.median(windows, axis=1).astype(data.dtype)


def rle_collapse(labels, remove_background=True, background_class=0):
    """
    Collapses consecutive identical labels into a single instance (Run-Length Encoding).

    Args:
        labels (list or np.array): Sequence of frame-wise labels.
        remove_background (bool): If True, removes instances of background_class.
        background_class (int): The ID representing the background/null class.

    Returns:
        list: Ordered list of gesture IDs.
    """
    if len(labels) == 0:
        return []

    collapsed = [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            collapsed.append(labels[i])

    if remove_background:
        collapsed = [x for x in collapsed if x != background_class]

    return collapsed


def decode_predictions(frame_probs, kernel_size=7, background_class=0):
    """
    Decodes frame-wise probabilities into a sequence of gestures.

    Args:
        frame_probs (np.array): Array of shape (T, NumClasses) containing probabilities or logits.
        kernel_size (int): Kernel size for median filtering.
        background_class (int): Class ID to filter out.

    Returns:
        list: Predicted sequence of gesture IDs.
    """
    # 1. Argmax to get class indices
    frame_preds = np.argmax(frame_probs, axis=1)

    # 2. Median Filter to smooth noise
    smoothed_preds = median_filter_1d(frame_preds, kernel_size=kernel_size)

    # 3. RLE Collapse to get sequence
    sequence = rle_collapse(
        smoothed_preds, remove_background=True, background_class=background_class
    )

    return sequence


def save_submission(predictions_dict, filename="submission.csv"):
    """
    Saves predictions to a CSV file in the required format.

    Format:
    Id,Sequence
    Sample00001,2 12 3

    Args:
        predictions_dict (dict): Dictionary mapping sample_id (str) to predicted sequence (list of ints).
        filename (str): Output filename.
    """
    if os.path.dirname(filename):
        output_path = filename
    else:
        output_path = os.path.join(SUBMISSION_DIR, filename)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        # Write Header
        f.write("Id,Sequence\n")

        for sample_id, sequence in predictions_dict.items():
            # Clean ID: Sample00300 -> 300
            clean_id = sample_id
            if clean_id.startswith("Sample"):
                clean_id = clean_id.replace("Sample", "")

            # Remove leading zeros by converting to int
            try:
                clean_id = str(int(clean_id))
            except ValueError:
                pass  # Keep original if not an integer

            # Convert sequence list to space-separated string "2 12 3"
            seq_str = " ".join(map(str, sequence))

            # Write line: Id,Sequence
            line = f"{clean_id},{seq_str}\n"
            f.write(line)

    print(f"Submission saved to {output_path}")
