import os
import sys
import random
import json
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name="experiment", log_file=None, level=logging.INFO):
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

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


def parse_labels(labels_str):
    """
    Parses the JSON string from the metadata CSV 'labels' column.

    Args:
        labels_str (str): JSON formatted string containing label data.

    Returns:
        list: List of dictionaries with label info, or empty list if parsing fails.
    """
    try:
        if isinstance(labels_str, str):
            return json.loads(labels_str)
        return []
    except Exception as e:
        # Silent failure preferred for utility functions used in loops,
        # but could log if logger was passed.
        return []


def get_target_sequence(labels_list):
    """
    Extracts the ground truth sequence of gesture IDs from the parsed labels list.
    Sorts by start frame to ensure correct order.

    Args:
        labels_list (list): List of dictionaries containing 'id', 'begin', 'end'.

    Returns:
        list[int]: Ordered list of gesture IDs.
    """
    if not labels_list:
        return []

    # Sort by 'begin' frame to ensure temporal order
    sorted_labels = sorted(labels_list, key=lambda x: x["begin"])

    # Extract IDs
    return [int(item["id"]) for item in sorted_labels]


def rle_encode_predictions(frame_predictions, background_id=Config.BACKGROUND_CLASS_ID):
    """
    Converts frame-wise predictions into a sequence of gesture IDs.
    1. Collapses consecutive identical labels (Run-Length Encoding).
    2. Removes the background class.

    Args:
        frame_predictions (list or np.array): List of predicted class IDs per frame.
        background_id (int): The ID representing the background/null class.

    Returns:
        list[int]: The sequence of recognized gestures.
    """
    if len(frame_predictions) == 0:
        return []

    # Collapse consecutive duplicates
    collapsed = []
    # Handle numpy arrays or lists
    preds = np.array(frame_predictions).flatten()

    if len(preds) > 0:
        collapsed.append(preds[0])
        for i in range(1, len(preds)):
            if preds[i] != preds[i - 1]:
                collapsed.append(preds[i])

    # Filter out background
    final_sequence = [int(x) for x in collapsed if x != background_id]

    return final_sequence


def compute_levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences using Dynamic Programming.

    Args:
        seq1 (list): First sequence.
        seq2 (list): Second sequence.

    Returns:
        float: The Levenshtein distance.
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


def evaluate_predictions(predictions, targets):
    """
    Computes the normalized Levenshtein distance score (Error Rate).
    Score = Total Levenshtein Distance / Total Number of Ground Truth Gestures.

    Args:
        predictions: List of predicted sequences (List[List[int]])
        targets: List of ground truth sequences (List[List[int]])

    Returns:
        float: The error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        dist = compute_levenshtein_distance(pred, target)
        total_distance += dist
        total_length += len(target)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def save_cache_data(data, filename):
    """
    Helper to save data to the cache directory defined in Config.
    Uses numpy compressed format (.npz).

    Args:
        data: The data object to save (numpy array or dict).
        filename (str): The filename (e.g., 'train_features.npz').
    """
    cache_path = os.path.join(Config.CACHE_DIR, filename)
    try:
        # If data is a dictionary, save as kwargs, else as 'data'
        if isinstance(data, dict):
            np.savez_compressed(cache_path, **data)
        else:
            np.savez_compressed(cache_path, data=data)
    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")


def load_cache_data(filename):
    """
    Helper to load data from the cache directory.

    Args:
        filename (str): The filename to load.

    Returns:
        The loaded data object, or None if not found/error.
    """
    cache_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(cache_path):
        try:
            # Allow pickle is often needed for object arrays, though we try to avoid pickle
            # strict numpy arrays don't need it.
            with np.load(cache_path, allow_pickle=True) as loaded:
                # If it was saved as a single array under 'data'
                if "data" in loaded.files and len(loaded.files) == 1:
                    return loaded["data"]
                # Otherwise return the NpzFile object-like dict
                return dict(loaded)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")
            return None
    return None
