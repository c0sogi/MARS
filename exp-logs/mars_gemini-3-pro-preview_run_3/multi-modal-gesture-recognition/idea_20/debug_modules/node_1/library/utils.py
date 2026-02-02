import os
import sys
import logging
import numpy as np
import nltk
from library.config import BACKGROUND_CLASS_ID


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to both the console and a file.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GestureRecognition")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def decode_predictions(frame_predictions):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using
    Run-Length Encoding (RLE) and background filtering.

    Args:
        frame_predictions (np.ndarray): Array of shape (T,) containing class indices
                                        or (T, C) containing probabilities.

    Returns:
        list: A list of integer gesture IDs (excluding background).
    """
    # If input is probabilities (2D), take argmax
    if frame_predictions.ndim > 1:
        labels = np.argmax(frame_predictions, axis=1)
    else:
        labels = frame_predictions

    if len(labels) == 0:
        return []

    # Run-Length Encoding: Collapse consecutive duplicates
    unique_consecutive = [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            unique_consecutive.append(labels[i])

    # Filter out background class
    final_sequence = [int(x) for x in unique_consecutive if x != BACKGROUND_CLASS_ID]

    return final_sequence


def compute_levenshtein(predictions, targets):
    """
    Computes the normalized Levenshtein distance (Error Rate).

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        targets (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The Levenshtein score (Total Distance / Total Ground Truth Gestures).
    """
    total_distance = 0
    total_truth_gestures = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure sequences are lists
        p_seq = list(pred_seq)
        t_seq = list(target_seq)

        # Compute edit distance
        dist = nltk.edit_distance(p_seq, t_seq)

        total_distance += dist
        total_truth_gestures += len(t_seq)

    # Avoid division by zero
    if total_truth_gestures == 0:
        return 0.0

    return total_distance / total_truth_gestures


def generate_submission_file(predictions, sample_ids, output_path):
    """
    Generates the submission CSV file.

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        sample_ids (list of str): List of sample IDs corresponding to predictions.
        output_path (str): Path to save the CSV file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for sid, pred_seq in zip(sample_ids, predictions):
            # Format: SessionID,Label1,Label2,...
            # Convert integers to strings
            str_preds = [str(p) for p in pred_seq]

            # Join with commas
            line_content = [str(sid)] + str_preds
            line = ",".join(line_content)

            f.write(line + "\n")
