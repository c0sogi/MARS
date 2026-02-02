import os
import sys
import random
import logging
import numpy as np
import torch
from scipy.ndimage import median_filter
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


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler_console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler_file = logging.FileHandler(log_file)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    return logger


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.
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


def compute_levenshtein_score(predicted_sequences, target_sequences):
    """
    Computes the competition metric: Sum of Levenshtein distances / Total target gestures.

    Args:
        predicted_sequences: List of lists containing predicted gesture IDs.
        target_sequences: List of lists containing ground truth gesture IDs.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_gestures = 0

    for pred, targ in zip(predicted_sequences, target_sequences):
        dist = levenshtein_distance(pred, targ)
        total_distance += dist
        total_gestures += len(targ)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def post_process_predictions(outputs, median_window=Config.MEDIAN_WINDOW_SIZE):
    """
    Converts raw model outputs (logits) into a sequence of gesture IDs.

    Steps:
    1. Argmax to get frame-wise labels.
    2. Median filtering with nearest-neighbor padding to smooth jitter.
    3. Collapse repeats and remove background class (0).

    Args:
        outputs: np.ndarray or torch.Tensor of shape (Batch, Classes, Time) or (Batch, Time, Classes).
        median_window: Kernel size for the median filter.

    Returns:
        List of lists, where each inner list is a sequence of gesture IDs (ints).
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()

    # Ensure shape is (Batch, Time, Classes) for argmax along last axis
    # If input is (Batch, Classes, Time), transpose it
    if outputs.ndim == 3 and outputs.shape[1] == Config.NUM_CLASSES:
        outputs = np.transpose(outputs, (0, 2, 1))

    # Get frame-wise predictions
    frame_preds = np.argmax(outputs, axis=-1)  # Shape: (Batch, Time)

    final_sequences = []

    for i in range(frame_preds.shape[0]):
        raw_seq = frame_preds[i]

        # Apply median filter with nearest padding to preserve boundaries
        # mode='nearest' replicates the edge values
        smoothed_seq = median_filter(raw_seq, size=median_window, mode="nearest")

        # Decode: Collapse repeats and remove background (0)
        decoded_seq = []
        prev_label = -1

        for label in smoothed_seq:
            if label != prev_label:
                if label != 0:  # Assuming 0 is background
                    decoded_seq.append(int(label))
                prev_label = label

        final_sequences.append(decoded_seq)

    return final_sequences
