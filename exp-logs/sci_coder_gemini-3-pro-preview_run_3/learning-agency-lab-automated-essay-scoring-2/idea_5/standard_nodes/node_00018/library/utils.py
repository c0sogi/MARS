import os
import random
import numpy as np
import torch
import logging
import hashlib
import json
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across random, numpy, and torch libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    This function handles continuous predictions by clipping them to the valid
    score range [1, 6] and rounding them to the nearest integer before
    computing the metric.

    Args:
        y_true: Array-like of true target values (integers).
        y_pred: Array-like of predicted values (floats or integers).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred)

    # If predictions are floats, clip to [1, 6] and round to nearest integer
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(np.clip(y_pred, 1, 6)).astype(int)
    else:
        # Ensure integer type for safety
        y_pred = np.array(y_pred, dtype=int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_logger(filename: str = "train.log"):
    """
    Initializes and returns a logger that outputs to both a file and the console.

    Args:
        filename (str): The path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    This is used for cache invalidation to ensure data is re-processed
    if parameters change.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: MD5 hash string.
    """
    # Use json.dumps with sort_keys=True for deterministic ordering.
    # default=str handles non-serializable objects like torch.device or pathlib.Path
    try:
        encoded = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        # Fallback if json serialization fails completely
        encoded = str(sorted(config_dict.items())).encode("utf-8")

    return hashlib.md5(encoded).hexdigest()
