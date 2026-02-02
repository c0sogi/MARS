import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        seq1 (list): First sequence of labels.
        seq2 (list): Second sequence of labels.

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
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein_ratio(preds_list, targets_list):
    """
    Computes the competition metric: Total Levenshtein Distance / Total True Gestures.

    Args:
        preds_list (list of lists): Predicted sequences.
        targets_list (list of lists): Ground truth sequences.

    Returns:
        float: The error rate.
    """
    total_distance = 0
    total_length = 0

    for p, t in zip(preds_list, targets_list):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def rle_decode(predictions, background_label=config.BACKGROUND_LABEL, min_duration=5):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using Run-Length Encoding logic.
    Filters out background class and short segments.

    Args:
        predictions (np.ndarray or list): Frame-wise class indices.
        background_label (int): The label index for background/no-gesture.
        min_duration (int): Minimum number of frames for a gesture to be considered valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # Identify changes in values
    # Pad with a different value at the end to ensure the last run is captured
    padded_preds = np.concatenate([predictions, [predictions[-1] + 1]])

    # Find indices where values change
    change_indices = np.where(padded_preds[:-1] != padded_preds[1:])[0] + 1

    decoded_sequence = []
    start_idx = 0

    for end_idx in change_indices:
        label = predictions[start_idx]
        duration = end_idx - start_idx

        # Filter logic
        if label != background_label and duration >= min_duration:
            decoded_sequence.append(int(label))

        start_idx = end_idx

    return decoded_sequence


def save_submission(predictions_dict, output_path):
    """
    Saves predictions to a CSV file in the submission format.

    Args:
        predictions_dict (dict): Dictionary mapping sample_id (str) to list of gesture IDs (list[int]).
        output_path (str): Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for sample_id, pred_sequence in predictions_dict.items():
            # Format: SessionID,Label1,Label2,...
            # If sequence is empty, just SessionID (or SessionID, depending on strict requirements,
            # but usually CSV implies comma separation. Prompt example: Session00001,2,12,3)

            line = sample_id
            if pred_sequence:
                line += "," + ",".join(map(str, pred_sequence))
            else:
                # If no gestures detected, we leave it as just the ID or ID, (empty)
                # Based on prompt "Session00001,2,12,3", if empty likely just "Session00001"
                pass

            f.write(line + "\n")
