import os
import sys
import random
import numpy as np
import logging
import hashlib
import json
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def setup_logging(log_filename="pipeline.log"):
    """
    Configures the logging system to write to a file and stdout.
    """
    log_path = os.path.join(Config.WORKING_DIR, log_filename)

    # Create a custom logger
    logger = logging.getLogger("NFL_Contact_Detection")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_path, mode="w")

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info(f"Logging set up. Log file: {log_path}")
    return logger


def seed_everything(seed=42):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch is listed in installed packages, so we seed it if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def generate_config_hash(params_dict):
    """
    Generates a deterministic MD5 hash from a dictionary of parameters.
    Used for parameter-aware caching.

    Args:
        params_dict (dict): Dictionary of configuration parameters.

    Returns:
        str: Hex digest of the hash.
    """
    # Filter out keys that shouldn't affect the hash if necessary,
    # but generally we want all config params to trigger a re-compute.
    # We sort keys to ensure deterministic ordering.

    # Helper to handle non-serializable objects (like classes) by converting to string
    def default_serializer(obj):
        return str(obj)

    params_str = json.dumps(params_dict, sort_keys=True, default=default_serializer)
    hash_obj = hashlib.md5(params_str.encode("utf-8"))
    return hash_obj.hexdigest()


def get_cache_path(base_filename, params_hash=None, ext=".parquet"):
    """
    Constructs the relative filename for a cache file.

    Args:
        base_filename (str): The core name of the file (e.g., 'train_features').
        params_hash (str, optional): The hash string to append.
        ext (str): The file extension.

    Returns:
        str: The relative filename.
    """
    if params_hash:
        filename = f"{base_filename}_{params_hash}{ext}"
    else:
        filename = f"{base_filename}{ext}"

    return filename


def save_to_parquet(df, filename):
    """
    Saves a pandas DataFrame to a parquet file in the working directory.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    df.to_parquet(path, index=False)
    # Using print/logging here might be redundant if the caller logs,
    # but it confirms the action.
    # logging.getLogger("NFL_Contact_Detection").info(f"Saved parquet to {path}")


def load_from_parquet(filename):
    """
    Loads a pandas DataFrame from a parquet file in the working directory.
    Returns None if file does not exist.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_to_npy(arr, filename):
    """
    Saves a numpy array to an .npy file in the working directory.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    np.save(path, arr)


def load_from_npy(filename):
    """
    Loads a numpy array from an .npy file in the working directory.
    Returns None if file does not exist.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    return None


def check_cache_exists(filename):
    """
    Checks if a specific file exists in the working directory.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    return os.path.exists(path)
