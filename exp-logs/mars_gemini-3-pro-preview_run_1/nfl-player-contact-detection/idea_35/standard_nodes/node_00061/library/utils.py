import os
import sys
import random
import numpy as np
import logging
import hashlib
import json
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logger(name: str = "experiment", log_file: str = None, level=logging.INFO):
    """
    Configures a logger to output to console and optionally a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, no file logging is performed.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_cache_path(base_path: str, params: dict) -> str:
    """
    Generates a unique cache file path based on the provided parameters.
    This ensures that if hyperparameters change, a new cache file is created/used.

    Args:
        base_path (str): The base path defined in Config (e.g., 'working/features.parquet').
        params (dict): A dictionary of parameters that affect the content of this file.
                       This should include things like window sizes, feature lists, etc.

    Returns:
        str: A modified path including a hash of the parameters
             (e.g., 'working/features_a1b2c3d4.parquet').
    """
    if params is None:
        return base_path

    # Serialize parameters to a JSON string with sorted keys to ensure determinism
    # We use a custom encoder or string conversion to handle non-serializable types if necessary,
    # but for config params (int, float, str, list), default json is usually fine.
    try:
        param_str = json.dumps(params, sort_keys=True, default=str)
    except TypeError:
        # Fallback for complex objects
        param_str = str(sorted(params.items()))

    # Compute MD5 hash
    param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()

    # Split base path
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)

    # Construct new filename
    new_filename = f"{name}_{param_hash}{ext}"

    return os.path.join(directory, new_filename)


def ensure_dir(file_path: str):
    """
    Ensures the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
