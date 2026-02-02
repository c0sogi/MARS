import os
import sys
import random
import numpy as np
import torch
import hashlib
import json
import logging
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def generate_content_hash(content):
    """
    Generates an MD5 hash for a given content (dict, list, string).
    Useful for creating unique cache filenames based on configuration parameters.

    Args:
        content: The object to hash (dict, list, tuple, string, or number).

    Returns:
        str: The hexadecimal MD5 hash digest.
    """
    try:
        if isinstance(content, dict):
            # Sort keys to ensure consistent ordering for hashing
            encoded = json.dumps(content, sort_keys=True).encode("utf-8")
        elif isinstance(content, (list, tuple)):
            encoded = json.dumps(content).encode("utf-8")
        else:
            encoded = str(content).encode("utf-8")
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        encoded = str(content).encode("utf-8")

    return hashlib.md5(encoded).hexdigest()


def setup_logger(name="nfl_contact_detection", log_file="execution.log"):
    """
    Sets up a logger that outputs to both console and a file in the working directory.

    Args:
        name (str): The name of the logger.
        log_file (str): The filename for the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    # Ensure working directory exists (Config handles this, but safety check doesn't hurt)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    log_path = os.path.join(Config.WORKING_DIR, log_file)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
