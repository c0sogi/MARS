import os
import numpy as np
import itertools
from scipy.signal import medfilt
from nltk import edit_distance
from library.config import set_seed


def decode_predictions(frame_probs, filter_size=7):
    """
    Decodes frame-wise probabilities or indices into a sequence of gesture IDs.

    Applies argmax (if probabilities are provided), Median Filtering,
    collapses consecutive duplicates (Run-Length Encoding), and removes the background class (0).

    Args:
        frame_probs (np.ndarray): Array of shape (T, NumClasses) containing probabilities/logits,
                                  or shape (T,) containing class indices.
        filter_size (int): Kernel size for median filtering.

    Returns:
        list: Ordered list of recognized gesture IDs (integers), excluding background.
    """
    # Convert to numpy if it's a tensor (just in case, though numpy is expected)
    if hasattr(frame_probs, "cpu"):
        frame_probs = frame_probs.cpu().numpy()

    # If input is probabilities (2D), take argmax to get indices
    if frame_probs.ndim > 1:
        frame_preds = np.argmax(frame_probs, axis=-1)
    else:
        frame_preds = frame_probs

    # Median Filter to smooth predictions
    frame_preds = medfilt(frame_preds, kernel_size=filter_size).astype(int)

    # Run-Length Encoding: collapse consecutive duplicates
    # e.g., [0, 0, 1, 1, 0, 2, 2] -> [0, 1, 0, 2]
    collapsed = [k for k, g in itertools.groupby(frame_preds)]

    # Filter out background (class 0)
    final_sequence = [int(x) for x in collapsed if x != 0]

    return final_sequence


def compute_levenshtein_ratio(predictions, ground_truths):
    """
    Computes the Levenshtein ratio metric: Total Edit Distance / Total Ground Truth Gestures.

    Args:
        predictions (list of list): List of predicted gesture ID sequences.
        ground_truths (list of list): List of ground truth gesture ID sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        # Ensure inputs are lists
        p = list(pred)
        t = list(truth)

        # Calculate Levenshtein distance
        dist = edit_distance(p, t)

        total_distance += dist
        total_length += len(t)

    # Avoid division by zero
    if total_length == 0:
        return 0.0

    return total_distance / total_length


def save_submission(predictions, sample_ids, output_path):
    """
    Saves predictions to a CSV file in the required submission format.

    Format: Id,Sequence
    Where Sequence is a space-separated list of labels.

    Args:
        predictions (list of list): List of predicted gesture sequences.
        sample_ids (list): List of session IDs corresponding to predictions.
        output_path (str): Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    # Add Header
    rows.append("Id,Sequence")

    for sid, pred_seq in zip(sample_ids, predictions):
        # Convert integers to strings
        pred_str_list = [str(p) for p in pred_seq]

        # Join predictions with space to create a single column for the sequence
        # Cite debug_lesson_1: Prevent Ragged CSV Errors by Using Non-Conflicting Delimiters
        pred_string = " ".join(pred_str_list)

        # Construct the line: SessionID,SequenceString
        line_str = f"{sid},{pred_string}"
        rows.append(line_str)

    # Write to file
    with open(output_path, "w") as f:
        for row in rows:
            f.write(row + "\n")
