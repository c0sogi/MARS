import os
import random
import numpy as np
import pandas as pd
import gc
import torch
from library.config import Config


def seed_everything(seed=None):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def reduce_mem_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    # print(f'Memory usage optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


class CacheManager:
    """
    Manages caching of intermediate data artifacts (Parquet/NPY) to the working directory.
    Implements strict logic to avoid re-computation and forbids pickle usage.
    """

    def __init__(self):
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_path(self, filename):
        """Constructs the full path for a cached file."""
        return os.path.join(self.cache_dir, filename)

    def exists(self, filename):
        """Checks if a file exists in the cache."""
        return os.path.exists(self.get_path(filename))

    def save(self, data, filename):
        """
        Saves data to the cache directory.

        Args:
            data: The data object (pd.DataFrame or np.ndarray).
            filename (str): The name of the file. Must end in .parquet or .npy.
        """
        filepath = self.get_path(filename)

        if filename.endswith(".parquet"):
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError(
                    "Data must be a pandas DataFrame for .parquet extension."
                )
        elif filename.endswith(".npy"):
            if isinstance(data, np.ndarray):
                np.save(filepath, data)
            else:
                raise ValueError("Data must be a numpy array for .npy extension.")
        else:
            raise ValueError(
                f"Unsupported file extension for {filename}. Use .parquet or .npy."
            )

        # print(f"Saved cache: {filepath}")

    def load(self, filename):
        """
        Loads data from the cache directory.

        Args:
            filename (str): The name of the file to load.

        Returns:
            The loaded data (pd.DataFrame or np.ndarray), or None if loading fails.
        """
        filepath = self.get_path(filename)

        if not os.path.exists(filepath):
            return None

        try:
            if filename.endswith(".parquet"):
                return pd.read_parquet(filepath)
            elif filename.endswith(".npy"):
                return np.load(filepath)
            else:
                raise ValueError(
                    f"Unsupported file extension for {filename}. Use .parquet or .npy."
                )
        except Exception as e:
            print(f"Error loading cache {filename}: {e}")
            return None

    def clear(self, filename=None):
        """
        Clears a specific file or the entire cache directory.
        """
        if filename:
            filepath = self.get_path(filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        else:
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
