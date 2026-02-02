import os
import sys
import logging
import random
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from library.config import WORKING_DIR, SEED


def setup_logger(log_file_path=None):
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        log_file_path (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("NFL_Contact_Detection")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def seed_everything(seed=SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CacheManager:
    """
    Manages caching of DataFrames to Parquet files to avoid redundant computation.
    Uses parameter hashing to ensure cache validity.
    """

    def __init__(self, cache_dir=WORKING_DIR):
        """
        Initialize the CacheManager.

        Args:
            cache_dir (str): Directory where cache files will be stored.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, prefix, params):
        """
        Generates a unique cache file path based on a prefix and a dictionary of parameters.

        Args:
            prefix (str): Identifier for the data (e.g., 'train_features_tier1').
            params (dict): Dictionary of parameters affecting the data generation.

        Returns:
            str: Full path to the cache file.
        """
        # Serialize params to a JSON string with sorted keys for consistency
        param_str = json.dumps(params, sort_keys=True, default=str)

        # Generate MD5 hash
        param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()

        # Construct filename
        filename = f"{prefix}_{param_hash}.parquet"
        return os.path.join(self.cache_dir, filename)

    def load(self, path):
        """
        Loads a DataFrame from a Parquet file if it exists.

        Args:
            path (str): Path to the cache file.

        Returns:
            pd.DataFrame or None: The loaded DataFrame, or None if file doesn't exist.
        """
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                return df
            except Exception as e:
                print(f"Failed to load cache from {path}: {e}")
                return None
        return None

    def save(self, df, path):
        """
        Saves a DataFrame to a Parquet file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            path (str): Path to the destination file.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"Failed to save cache to {path}: {e}")

    def clear(self):
        """
        Clears all files in the cache directory. Use with caution.
        """
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
