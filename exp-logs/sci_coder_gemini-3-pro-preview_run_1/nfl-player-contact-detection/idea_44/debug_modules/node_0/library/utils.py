import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import hashlib
import json
import joblib
from typing import Any, Dict, Optional, Union

from library.config import Config


def setup_logging(log_file_path: str = "execution.log") -> logging.Logger:
    """
    Configures logging to both a file and the console.

    Args:
        log_file_path: Path to the log file.

    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("KARP_AM")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def seed_everything(seed: int = 42):
    """
    Sets seeds for random, numpy, and os environments to ensure reproducibility.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Note: Torch is not imported here as per instructions, but if used later,
    # it should be seeded in the respective module.


class CacheManager:
    """
    Manages parameter-aware caching of intermediate results (DataFrames, Models, Arrays).
    Uses hashing of configuration dictionaries to ensure cache validity.
    """

    def __init__(self, cache_dir: str = Config.WORKING_DIR):
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory where cached files will be stored.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """
        Generates a unique filename key based on a prefix and a dictionary of parameters.

        Args:
            prefix: A string identifier for the file (e.g., 'features_train').
            params: A dictionary of parameters influencing the generation of this data.

        Returns:
            A string filename (without extension).
        """
        # Sort keys to ensure deterministic ordering
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()
        return f"{prefix}_{param_hash}"

    def save(self, data: Any, filename: str) -> str:
        """
        Saves data to the cache directory. Automatically detects format based on extension.

        Args:
            data: The object to save (DataFrame, numpy array, or generic object).
            filename: The full filename including extension (e.g., 'data_hash.parquet').

        Returns:
            The full path to the saved file.
        """
        file_path = os.path.join(self.cache_dir, filename)

        if filename.endswith(".parquet") and isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, index=False)
        elif filename.endswith(".npy") and isinstance(data, np.ndarray):
            np.save(file_path, data)
        elif filename.endswith(".joblib"):
            joblib.dump(data, file_path)
        else:
            # Fallback for generic objects if extension not strictly matched above
            # but usually we want strict control.
            if isinstance(data, pd.DataFrame):
                # Enforce parquet for dataframes if extension missing
                file_path += ".parquet"
                data.to_parquet(file_path, index=False)
            else:
                # Default to joblib for models/lists/dicts
                if not filename.endswith(".joblib"):
                    file_path += ".joblib"
                joblib.dump(data, file_path)

        return file_path

    def load(self, filename: str) -> Optional[Any]:
        """
        Loads data from the cache directory if it exists.

        Args:
            filename: The full filename including extension.

        Returns:
            The loaded data object, or None if the file does not exist.
        """
        file_path = os.path.join(self.cache_dir, filename)

        if not os.path.exists(file_path):
            return None

        try:
            if filename.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif filename.endswith(".npy"):
                return np.load(file_path)
            elif filename.endswith(".joblib"):
                return joblib.load(file_path)
            else:
                # Attempt joblib load for unspecified extensions
                return joblib.load(file_path)
        except Exception as e:
            print(f"Warning: Failed to load cache file {file_path}. Error: {e}")
            return None

    def exists(self, filename: str) -> bool:
        """Checks if a file exists in the cache."""
        return os.path.exists(os.path.join(self.cache_dir, filename))
