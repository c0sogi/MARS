import os
import random
import numpy as np
import torch
import pandas as pd
import logging
import hashlib
import json
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed: int = 42):
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
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): True labels (integers 1-6).
        y_pred (array-like): Predicted scores (continuous or integers).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to valid range [1, 6] and round to nearest integer
    y_pred = np.clip(y_pred, 1, 6).round().astype(int)
    y_true = y_true.astype(int)

    # Compute QWK
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return qwk


def get_logger(name: str = "EssayScoring"):
    """
    Creates and configures a logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Stream Handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class CacheManager:
    """
    Manages caching of intermediate results (like OOF embeddings) using hash-based validation.
    Ensures data consistency by hashing configuration parameters.
    """

    def __init__(self, cache_dir: str = Config.cache_dir):
        """
        Initialize the CacheManager.

        Args:
            cache_dir (str): Directory where cached files will be stored.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = get_logger("CacheManager")

    def get_config_hash(self, config_dict: dict) -> str:
        """
        Generates a simplified MD5 hash from a configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.

        Returns:
            str: A short hash string.
        """
        # Sort keys to ensure consistent ordering
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:10]

    def get_file_path(self, prefix: str, config_hash: str, ext: str) -> str:
        """
        Constructs the file path for a cached item.

        Args:
            prefix (str): Identifier for the file (e.g., 'oof_preds').
            config_hash (str): Hash of the configuration.
            ext (str): File extension (e.g., 'npy', 'parquet').

        Returns:
            str: Full path to the file.
        """
        filename = f"{prefix}_{config_hash}.{ext}"
        return os.path.join(self.cache_dir, filename)

    def save(self, data, prefix: str, config_dict: dict = None, ext: str = "npy"):
        """
        Saves data to the cache directory.

        Args:
            data: The data to save (numpy array or pandas DataFrame).
            prefix (str): Identifier for the file.
            config_dict (dict, optional): Configuration to hash. If None, no hash is appended.
            ext (str): Extension ('npy' or 'parquet').
        """
        if config_dict:
            config_hash = self.get_config_hash(config_dict)
            path = self.get_file_path(prefix, config_hash, ext)
        else:
            path = os.path.join(self.cache_dir, f"{prefix}.{ext}")

        try:
            if ext == "npy":
                np.save(path, data)
            elif ext == "parquet":
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(path, index=False)
                else:
                    raise ValueError("Data must be a DataFrame for parquet format.")
            else:
                raise ValueError(f"Unsupported extension: {ext}")

            # self.logger.info(f"Saved cached file to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save cache to {path}: {e}")
            raise e

    def load(self, prefix: str, config_dict: dict = None, ext: str = "npy"):
        """
        Loads data from the cache directory if it exists.

        Args:
            prefix (str): Identifier for the file.
            config_dict (dict, optional): Configuration to hash.
            ext (str): Extension ('npy' or 'parquet').

        Returns:
            The loaded data or None if not found.
        """
        if config_dict:
            config_hash = self.get_config_hash(config_dict)
            path = self.get_file_path(prefix, config_hash, ext)
        else:
            path = os.path.join(self.cache_dir, f"{prefix}.{ext}")

        if not os.path.exists(path):
            return None

        try:
            if ext == "npy":
                data = np.load(path)
            elif ext == "parquet":
                data = pd.read_parquet(path)
            else:
                return None

            # self.logger.info(f"Loaded cached file from {path}")
            return data

        except Exception as e:
            self.logger.warning(f"Failed to load cache from {path}: {e}")
            return None

    def exists(self, prefix: str, config_dict: dict = None, ext: str = "npy") -> bool:
        """
        Checks if a cached file exists.

        Args:
            prefix (str): Identifier.
            config_dict (dict, optional): Configuration to hash.
            ext (str): Extension.

        Returns:
            bool: True if file exists, False otherwise.
        """
        if config_dict:
            config_hash = self.get_config_hash(config_dict)
            path = self.get_file_path(prefix, config_hash, ext)
        else:
            path = os.path.join(self.cache_dir, f"{prefix}.{ext}")

        return os.path.exists(path)
