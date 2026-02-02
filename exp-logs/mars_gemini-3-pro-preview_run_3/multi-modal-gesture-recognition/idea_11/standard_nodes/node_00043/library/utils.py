import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_file=os.path.join(Config.WORKING_DIR, "training.log")):
    """
    Sets up a logger that writes to both the console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("BA_AKN_Logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def levenshtein_distance(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        seq1 (list): First sequence (prediction).
        seq2 (list): Second sequence (ground truth).

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


def compute_levenshtein(predictions, targets):
    """
    Computes the average Levenshtein error rate over a batch or dataset.
    Score = Sum(Distances) / Sum(GroundTruthLengths)

    Args:
        predictions (list of lists): List of predicted gesture ID sequences.
        targets (list of lists): List of ground truth gesture ID sequences.

    Returns:
        float: The computed error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, targ in zip(predictions, targets):
        dist = levenshtein_distance(pred, targ)
        total_distance += dist
        total_length += len(targ)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def collapse_predictions(frame_predictions):
    """
    Converts frame-wise class predictions into a sequence of gesture segments.
    Applies Run-Length Encoding (RLE) logic:
    1. Collapses consecutive identical labels.
    2. Removes the background label (0).

    Args:
        frame_predictions (np.ndarray or list): Array of frame-wise labels.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if isinstance(frame_predictions, torch.Tensor):
        frame_predictions = frame_predictions.cpu().numpy()

    collapsed = []
    last_label = -1

    for label in frame_predictions:
        label = int(label)
        if label != last_label:
            if label != Config.BACKGROUND_LABEL:
                collapsed.append(label)
            last_label = label

    return collapsed
