import os
import random
import numpy as np
import torch
import logging
import hashlib
import json
import sys
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str = None):
    """
    Initializes and configures a logger that outputs to both a file and the console.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("essay_scoring_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    if filename:
        # Ensure directory exists
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): True labels (scores).
        y_pred (array-like): Predicted scores.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to the valid range [1, 6] and round to nearest integer
    # This handles regression outputs that need to be converted to ordinal classes
    y_pred = np.clip(y_pred, 1, 6).round().astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_hash(obj):
    """
    Generates a deterministic MD5 hash for a given object (e.g., config dictionary).
    Useful for creating unique cache filenames.

    Args:
        obj: The object to hash.

    Returns:
        str: Hexadecimal MD5 hash string.
    """
    if isinstance(obj, (dict, list)):
        # Sort keys to ensure consistent ordering for dictionaries
        obj_str = json.dumps(obj, sort_keys=True)
    else:
        obj_str = str(obj)

    return hashlib.md5(obj_str.encode("utf-8")).hexdigest()
