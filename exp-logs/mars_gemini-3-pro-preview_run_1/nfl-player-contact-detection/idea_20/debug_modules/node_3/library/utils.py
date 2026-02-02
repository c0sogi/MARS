import os
import sys
import random
import logging
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from library.config import SEED, IDEA_DIR


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="rks_mte_logger", log_file=None):
    """
    Configures a logger with both stream (console) and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


class CacheManager:
    """
    Manages loading and saving of intermediate data (DataFrames or NumPy arrays)
    using parameter-aware hashing to ensure cache validity.
    """

    def __init__(self, cache_dir=IDEA_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_hash(self, params):
        """
        Generates an MD5 hash from a dictionary of parameters.
        Returns an empty string if params is None.
        """
        if params is None:
            return ""

        # Sort keys to ensure consistent string representation
        try:
            param_str = json.dumps(params, sort_keys=True, default=str)
        except TypeError:
            # Fallback for non-serializable objects
            param_str = str(sorted(params.items()))

        return hashlib.md5(param_str.encode("utf-8")).hexdigest()

    def _get_filepath(self, name, params, extension):
        """Constructs the full file path based on name, params hash, and extension."""
        hash_str = self._generate_hash(params)
        if hash_str:
            filename = f"{name}_{hash_str}{extension}"
        else:
            filename = f"{name}{extension}"
        return os.path.join(self.cache_dir, filename)

    def load(self, name, params=None):
        """
        Attempts to load cached data.
        Checks for .parquet (DataFrame) first, then .npy (NumPy array).
        Returns None if no cache is found.
        """
        # 1. Try Parquet (Pandas DataFrame)
        path_parquet = self._get_filepath(name, params, ".parquet")
        if os.path.exists(path_parquet):
            return pd.read_parquet(path_parquet)

        # 2. Try Numpy (ndarray)
        path_npy = self._get_filepath(name, params, ".npy")
        if os.path.exists(path_npy):
            return np.load(path_npy)

        return None

    def save(self, data, name, params=None):
        """
        Saves data to the cache directory.
        Uses .parquet for DataFrames and .npy for NumPy arrays.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            path = self._get_filepath(name, params, ".parquet")
            data.to_parquet(path, index=False)
        elif isinstance(data, np.ndarray):
            path = self._get_filepath(name, params, ".npy")
            np.save(path, data)
        else:
            raise ValueError(
                f"Unsupported data type for caching: {type(data)}. "
                "Only pd.DataFrame and np.ndarray are supported."
            )
