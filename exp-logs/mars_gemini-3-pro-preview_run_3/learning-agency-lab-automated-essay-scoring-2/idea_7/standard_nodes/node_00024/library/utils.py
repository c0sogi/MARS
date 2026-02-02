import os
import random
import numpy as np
import torch
import logging
import sys
import hashlib
import json
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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
    Computes the Quadratic Weighted Kappa (QWK) metric.

    Since the pipeline treats the score as a regression target, this function
    handles the necessary rounding and clipping to convert continuous predictions
    into the integer classes [1, 6] required by QWK.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted scores (continuous or discrete).

    Returns:
        float: The quadratic weighted kappa score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Round continuous predictions to nearest integer
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(y_pred)

    # Clip to valid score range [1, 6] and convert to integer
    y_pred = np.clip(y_pred, 1, 6).astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_logger(filename):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        filename (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is retrieved multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File Handler
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler for logger: {e}")

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_hash(config_dict):
    """
    Generates a deterministic MD5 hash from a configuration dictionary.
    Used for creating unique cache filenames based on model hyperparameters.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: Hexadecimal MD5 hash string.
    """

    def default(o):
        # Handle non-serializable objects by converting to string
        return str(o)

    # Sort keys to ensure deterministic output regardless of insertion order
    encoded = json.dumps(config_dict, sort_keys=True, default=default).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()
