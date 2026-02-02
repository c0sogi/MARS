import os
import sys
import logging
import random
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations can be slower, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file="execution.log"):
    """
    Configures a logger to write to both a file and the console (stdout).
    """
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("ECS-ME")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if function is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.
    Returns the float value directly.
    """
    return matthews_corrcoef(y_true, y_pred)


def get_config_hash(params_dict):
    """
    Generates a unique MD5 hash based on a dictionary of configuration parameters.
    Used to version cached files based on the hyperparameters used to create them.
    """
    # Sort keys to ensure dictionary order doesn't affect hash
    params_str = json.dumps(params_dict, sort_keys=True)
    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def save_to_parquet(df, filename):
    """
    Saves a DataFrame to a parquet file within the configured cache directory.
    Ensures the directory exists.
    """
    file_path = Config.get_cache_path(filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path, index=False)
    return file_path


def load_from_parquet(filename):
    """
    Loads a DataFrame from a parquet file within the configured cache directory.
    Returns None if the file does not exist.
    """
    file_path = Config.get_cache_path(filename)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None


def save_to_npy(arr, filename):
    """
    Saves a numpy array to an .npy file within the configured cache directory.
    """
    file_path = Config.get_cache_path(filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, arr)
    return file_path


def load_from_npy(filename):
    """
    Loads a numpy array from an .npy file within the configured cache directory.
    Returns None if the file does not exist.
    """
    file_path = Config.get_cache_path(filename)
    if os.path.exists(file_path):
        return np.load(file_path)
    return None


def execute_with_cache(filename, func, load_cached_data=True, *args, **kwargs):
    """
    Standardized caching logic wrapper.

    Logic:
    1. If load_cached_data is True, try to load from cache.
    2. If successful, return cached data.
    3. If not (or if load_cached_data is False), execute 'func', save result, and return.

    Args:
        filename (str): The name of the cache file (e.g., 'features_train_v1.parquet').
        func (callable): The function to compute the data if cache is missed.
        load_cached_data (bool): Whether to attempt loading from cache.
        *args, **kwargs: Arguments passed to 'func'.

    Returns:
        The data (DataFrame or Array).
    """
    logger = logging.getLogger("ECS-ME")

    # Determine loader/saver based on extension
    if filename.endswith(".parquet"):
        loader = load_from_parquet
        saver = save_to_parquet
    elif filename.endswith(".npy"):
        loader = load_from_npy
        saver = save_to_npy
    else:
        raise ValueError("Unsupported cache file format. Use .parquet or .npy")

    # 1. Try Load
    if load_cached_data:
        data = loader(filename)
        if data is not None:
            logger.info(f"Cache Hit: Loaded {filename}")
            return data
        else:
            logger.info(f"Cache Miss: {filename} not found. Computing...")
    else:
        logger.info(f"Cache Skip: Force re-computing {filename}...")

    # 2. Compute
    data = func(*args, **kwargs)

    # 3. Save
    saver(data, filename)
    logger.info(f"Saved computed data to {filename}")

    return data
