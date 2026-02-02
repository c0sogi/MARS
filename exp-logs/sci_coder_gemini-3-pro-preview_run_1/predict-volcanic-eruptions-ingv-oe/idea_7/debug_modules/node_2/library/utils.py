import os
import hashlib
import json
import pandas as pd
import numpy as np
from library.config import CACHE_DIR, seed_everything


class CacheManager:
    """
    Manages caching of DataFrames and NumPy arrays to disk using hashed filenames
    based on processing parameters.
    """

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_hash(self, params):
        """Generates an MD5 hash from a dictionary of parameters."""
        if not params:
            return ""
        # Sort keys to ensure consistent JSON string regardless of dict order
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()

    def _get_file_path(self, base_name, params=None, ext=None):
        """Constructs the full file path with hash and extension."""
        hash_str = self._generate_hash(params)

        filename = base_name
        if hash_str:
            filename = f"{filename}_{hash_str}"

        # Ensure extension is present
        if ext:
            if not filename.endswith(ext):
                filename += ext

        return os.path.join(self.cache_dir, filename)

    def exists(self, base_name, params=None, ext=None):
        """Checks if the cached file exists."""
        file_path = self._get_file_path(base_name, params, ext)
        return os.path.exists(file_path)

    def save(self, data, base_name, params=None, ext=None):
        """
        Saves data to the cache directory.

        Args:
            data: pandas DataFrame or numpy ndarray
            base_name: Identifier for the file
            params: Dictionary of parameters used to generate the data (for hashing)
            ext: File extension (optional, defaults to .parquet for DF, .npy for array)
        """
        if isinstance(data, pd.DataFrame):
            if ext is None:
                ext = ".parquet"
            file_path = self._get_file_path(base_name, params, ext)
            data.to_parquet(file_path, index=False)

        elif isinstance(data, np.ndarray):
            if ext is None:
                ext = ".npy"
            file_path = self._get_file_path(base_name, params, ext)
            np.save(file_path, data)

        else:
            raise ValueError(
                "CacheManager only supports pandas DataFrames and numpy ndarrays."
            )

    def load(self, base_name, params=None, ext=None):
        """
        Loads data from the cache directory.

        Args:
            base_name: Identifier for the file
            params: Dictionary of parameters used to generate the data
            ext: File extension

        Returns:
            The loaded data or None if file not found.
        """
        file_path = self._get_file_path(base_name, params, ext)

        if not os.path.exists(file_path):
            return None

        # Infer loader based on extension
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".npy"):
            return np.load(file_path)
        else:
            # Fallback if extension was not explicit in path but passed in arg
            if ext == ".parquet":
                return pd.read_parquet(file_path)
            elif ext == ".npy":
                return np.load(file_path)
            else:
                raise ValueError(f"Unsupported file extension for loading: {file_path}")
