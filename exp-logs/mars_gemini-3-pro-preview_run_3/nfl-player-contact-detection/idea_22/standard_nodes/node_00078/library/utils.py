import os
import sys
import random
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import Config from the provided library path
from library.config import Config

try:
    import torch
except ImportError:
    torch = None


def setup_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch (if available)
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Array-like of ground truth (correct) target values.
        y_pred: Array-like of predicted values.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are treated as binary for MCC if not already
    return matthews_corrcoef(y_true, y_pred)


class CacheManager:
    """
    Manages caching of intermediate data using hash-based invalidation.
    Stores data in Parquet (for DataFrames) or NPY (for Arrays) formats.
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_cache_id(self, config_dict, prefix="data"):
        """
        Generates a unique cache identifier string based on a configuration dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration parameters
                                (e.g., feature lists, lag offsets).
            prefix (str): Prefix for the filename.

        Returns:
            str: A unique identifier string (e.g., 'features_streamA_MD5HASH').
        """
        # Sort keys to ensure deterministic JSON string representation
        config_str = json.dumps(config_dict, sort_keys=True)

        # Compute MD5 hash of the configuration
        hash_obj = hashlib.md5(config_str.encode("utf-8"))
        hash_str = hash_obj.hexdigest()

        return f"{prefix}_{hash_str}"

    def get_file_path(self, cache_id, file_type):
        """
        Constructs the full file path for a given cache ID and type.
        """
        extension = "parquet" if file_type == "parquet" else "npy"
        return os.path.join(self.cache_dir, f"{cache_id}.{extension}")

    def load(self, cache_id, file_type="parquet"):
        """
        Attempts to load data from the cache.

        Args:
            cache_id (str): The unique identifier for the cached file.
            file_type (str): 'parquet' for pd.DataFrame or 'npy' for np.ndarray.

        Returns:
            The loaded data (pd.DataFrame or np.ndarray) if found, else None.
        """
        file_path = self.get_file_path(cache_id, file_type)

        if not os.path.exists(file_path):
            return None

        try:
            if file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "npy":
                return np.load(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            # In case of corruption or read error, return None to trigger re-computation
            print(f"Warning: Failed to load cache from {file_path}: {e}")
            return None

    def save(self, data, cache_id, file_type="parquet"):
        """
        Saves data to the cache.

        Args:
            data: The data to save (pd.DataFrame or np.ndarray).
            cache_id (str): The unique identifier for the file.
            file_type (str): 'parquet' for pd.DataFrame or 'npy' for np.ndarray.
        """
        file_path = self.get_file_path(cache_id, file_type)

        try:
            if file_type == "parquet":
                if not isinstance(data, pd.DataFrame):
                    raise ValueError(
                        "Data must be a pandas DataFrame for parquet format."
                    )
                data.to_parquet(file_path, index=False)
            elif file_type == "npy":
                if not isinstance(data, (np.ndarray, list)):
                    raise ValueError(
                        "Data must be a numpy array or list for npy format."
                    )
                np.save(file_path, data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            print(f"Error: Failed to save cache to {file_path}: {e}")
