import os
import random
import numpy as np
import pandas as pd
import hashlib
import json
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import WORKING_DIR


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


class CacheManager:
    """
    Manages hash-based caching for DataFrames (parquet) and Numpy arrays (npy).
    """

    def __init__(self, cache_dir=WORKING_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_hashed_filename(self, prefix, config, ext):
        """
        Generates a filename based on a prefix and a hash of the configuration.

        Args:
            prefix (str): Identifier for the file (e.g., 'features_stream_a').
            config (dict or list or str): Configuration object to hash.
            ext (str): File extension (e.g., 'parquet', 'npy').

        Returns:
            str: The full filename including the hash.
        """
        # Serialize config to a JSON string with sorted keys for consistency
        config_str = json.dumps(config, sort_keys=True, default=str)
        # Generate MD5 hash
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
        filename = f"{prefix}_{config_hash}.{ext}"
        return filename

    def get_path(self, filename):
        """Returns the full path for a given filename in the cache directory."""
        return os.path.join(self.cache_dir, filename)

    def exists(self, filename):
        """Checks if the file exists in the cache."""
        return os.path.exists(self.get_path(filename))

    def load(self, filename):
        """
        Loads data from the cache based on file extension.

        Args:
            filename (str): The name of the file to load.

        Returns:
            pd.DataFrame or np.ndarray: The loaded data, or None if loading fails.
        """
        file_path = self.get_path(filename)
        if not os.path.exists(file_path):
            return None

        try:
            if filename.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif filename.endswith(".npy"):
                return np.load(file_path)
            else:
                raise ValueError(f"Unsupported file extension for loading: {filename}")
        except Exception as e:
            print(f"Error loading cache file {filename}: {e}")
            return None

    def save(self, filename, data):
        """
        Saves data to the cache based on file extension.

        Args:
            filename (str): The name of the file to save.
            data: The pandas DataFrame or numpy array to save.
        """
        file_path = self.get_path(filename)
        try:
            if filename.endswith(".parquet"):
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, index=False)
                else:
                    raise TypeError(
                        "Data must be a pandas DataFrame for .parquet extension"
                    )
            elif filename.endswith(".npy"):
                if isinstance(data, np.ndarray):
                    np.save(file_path, data)
                else:
                    raise TypeError("Data must be a numpy array for .npy extension")
            else:
                raise ValueError(f"Unsupported file extension for saving: {filename}")
        except Exception as e:
            print(f"Error saving cache file {filename}: {e}")
