import os
import sys
import logging
import random
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import KADM_CONFIG


def setup_logger(name="kadm_logger", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance that outputs to stdout and optionally a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if they already exist
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed=42):
    """
    Sets seeds for reproducibility across random, numpy, and environment.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def _compute_config_hash(config_dict):
    """
    Computes a SHA256 hash of the configuration dictionary for caching purposes.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: Hex digest of the hash.
    """
    # Serialize config to JSON string with sorted keys for determinism
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def process_with_cache(
    func,
    cache_key,
    config_dict,
    load_cached_data=True,
    file_format="parquet",
    cache_dir=None,
    *args,
    **kwargs,
):
    """
    Wraps a processing function with parameter-aware caching logic.

    Args:
        func (callable): The function to execute if cache is missed.
        cache_key (str): Unique identifier for the cached file (e.g., 'train_features').
        config_dict (dict): Configuration parameters to hash.
        load_cached_data (bool): Whether to attempt loading from cache.
        file_format (str): 'parquet' or 'npy'.
        cache_dir (str, optional): Directory to save cache. Defaults to config if None.
        *args, **kwargs: Arguments passed to func.

    Returns:
        The processed data (loaded from cache or computed).
    """
    if cache_dir is None:
        cache_dir = KADM_CONFIG["paths"]["cache_dir"]

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Compute hash and filename
    config_hash = _compute_config_hash(config_dict)
    filename = f"{cache_key}_{config_hash}.{file_format}"
    file_path = os.path.join(cache_dir, filename)

    # Attempt to load from cache
    if load_cached_data and os.path.exists(file_path):
        print(f"Loading cached {cache_key} from {file_path}...")
        try:
            if file_format == "parquet":
                return pd.read_parquet(file_path)
            elif file_format == "npy":
                return np.load(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # Compute data if cache miss or load failed
    print(f"Computing {cache_key}...")
    data = func(*args, **kwargs)

    # Save to cache
    print(f"Saving {cache_key} to {file_path}...")
    if file_format == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False)
        else:
            raise TypeError("Data must be a pandas DataFrame for parquet format.")
    elif file_format == "npy":
        if isinstance(data, np.ndarray):
            np.save(file_path, data)
        else:
            raise TypeError("Data must be a numpy array for npy format.")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return data


def save_submission(df, path=None):
    """
    Saves the submission DataFrame to the specified path.

    Args:
        df (pd.DataFrame): The submission dataframe.
        path (str, optional): Path to save the CSV. Defaults to config path.
    """
    if path is None:
        path = KADM_CONFIG["paths"]["submission_output"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
