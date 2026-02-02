import os
import sys
import random
import numpy as np
import torch
import logging
import hashlib
import json
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main", filename: str = None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.
        filename (str, optional): The path to the log file. If None, uses Config.OUTPUT_LOG_DIR.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if filename is None:
        filename = os.path.join(Config.OUTPUT_LOG_DIR, f"{name}.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): True labels (1-6).
        y_pred (array-like): Predicted labels.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays and integers
    y_true = np.array(y_true, dtype=int)
    # Clip predictions to valid range [1, 6] and round
    y_pred = np.clip(np.round(y_pred), 1, 6).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_object_hash(obj) -> str:
    """
    Generates an MD5 hash for a given Python object (dict, list, string, etc.).
    Useful for creating unique cache filenames based on configuration/hyperparameters.

    Args:
        obj: The object to hash.

    Returns:
        str: The hexadecimal hash string.
    """
    if isinstance(obj, (dict, list, tuple)):
        # Convert to JSON string with sorted keys to ensure consistency
        obj_str = json.dumps(obj, sort_keys=True, default=str)
    else:
        obj_str = str(obj)

    return hashlib.md5(obj_str.encode("utf-8")).hexdigest()


def get_cache_path(base_name: str, config_obj=None) -> str:
    """
    Constructs a cache file path based on a base name and an optional configuration object.

    Args:
        base_name (str): The prefix for the filename (e.g., "oof_preds").
        config_obj: An object/dict to hash for versioning. If None, no hash is appended.

    Returns:
        str: The full path to the cache file.
    """
    if config_obj is not None:
        config_hash = get_object_hash(config_obj)
        filename = f"{base_name}_{config_hash}.npy"
    else:
        filename = f"{base_name}.npy"

    return os.path.join(Config.CACHE_DIR, filename)
