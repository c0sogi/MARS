import os
import sys
import random
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        seq1 (list or np.array): The first sequence (e.g., predicted gestures).
        seq2 (list or np.array): The second sequence (e.g., ground truth gestures).

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
                cost = 0
            else:
                cost = 1

            matrix[x, y] = min(
                matrix[x - 1, y] + 1,  # Deletion
                matrix[x, y - 1] + 1,  # Insertion
                matrix[x - 1, y - 1] + cost,  # Substitution
            )

    return matrix[size_x - 1, size_y - 1]


def run_length_encoding(predictions, min_length=5):
    """
    Converts frame-wise predictions to a sequence of gesture labels using Run-Length Encoding.

    Logic:
    1. Collapses consecutive identical labels.
    2. Removes background class (assumed to be 0).
    3. Filters out segments shorter than `min_length`.

    Args:
        predictions (list, np.array, or torch.Tensor): Frame-wise label predictions.
        min_length (int): Minimum number of frames for a gesture to be considered valid.

    Returns:
        list: A list of integer gesture IDs.
    """
    if len(predictions) == 0:
        return []

    # Ensure predictions is a numpy array or list
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    segments = []
    current_label = predictions[0]
    current_len = 1

    for i in range(1, len(predictions)):
        if predictions[i] == current_label:
            current_len += 1
        else:
            # End of a segment
            if current_label != 0 and current_len >= min_length:
                segments.append(int(current_label))

            # Start new segment
            current_label = predictions[i]
            current_len = 1

    # Handle the final segment
    if current_label != 0 and current_len >= min_length:
        segments.append(int(current_label))

    return segments


def compute_metric(predictions_list, targets_list):
    """
    Computes the normalized Levenshtein distance score for a batch/set of sequences.

    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures

    Args:
        predictions_list (list of lists): List of predicted gesture sequences.
        targets_list (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_dist = 0
    total_gestures = 0

    for pred, target in zip(predictions_list, targets_list):
        dist = compute_levenshtein(pred, target)
        total_dist += dist
        total_gestures += len(target)

    if total_gestures == 0:
        return 0.0 if total_dist == 0 else float("inf")

    return total_dist / total_gestures


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is reused
    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
