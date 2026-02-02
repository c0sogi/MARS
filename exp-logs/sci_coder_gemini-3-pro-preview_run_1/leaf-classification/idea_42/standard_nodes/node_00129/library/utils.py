import os
import random
import numpy as np
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and environment variables.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_float64(data):
    """
    Strictly enforces double-precision floating-point types for data arrays.
    This is critical for avoiding numerical instability and the 1e-7 metric floor
    in log-loss calculations.

    Args:
        data: Input data (numpy array, pandas DataFrame, pandas Series, or list).

    Returns:
        Data converted to np.float64 precision.
    """
    if isinstance(data, pd.DataFrame):
        # Convert all columns to float64
        return data.astype(Config.FLOAT_TYPE)
    elif isinstance(data, pd.Series):
        return data.astype(Config.FLOAT_TYPE)
    else:
        # Convert lists or existing arrays to float64 numpy array
        return np.array(data, dtype=Config.FLOAT_TYPE)


def save_to_cache(filename, data):
    """
    Saves data to the configured cache directory using appropriate formats.
    Enforces the use of Parquet for DataFrames and NPY for NumPy arrays,
    avoiding pickle as per requirements.

    Args:
        filename (str): The name of the file (extension will be appended if missing).
        data: The data object to save (pd.DataFrame or np.ndarray).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    filepath = os.path.join(Config.CACHE_DIR, filename)

    if isinstance(data, pd.DataFrame):
        if not filepath.endswith(".parquet"):
            filepath += ".parquet"
        data.to_parquet(filepath, index=False)
    elif isinstance(data, np.ndarray):
        if not filepath.endswith(".npy"):
            filepath += ".npy"
        np.save(filepath, data)
    else:
        raise ValueError(
            f"Unsupported data type for caching: {type(data)}. Use pd.DataFrame or np.ndarray."
        )


def load_from_cache(filename, expected_type="dataframe"):
    """
    Loads data from the configured cache directory.

    Args:
        filename (str): The name of the file (without extension is preferred).
        expected_type (str): 'dataframe' or 'numpy'. Defaults to 'dataframe'.

    Returns:
        The loaded data object, or None if the file does not exist.
    """
    base_path = os.path.join(Config.CACHE_DIR, filename)

    if expected_type == "dataframe":
        # Check for parquet extension
        path = base_path if base_path.endswith(".parquet") else base_path + ".parquet"
        if os.path.exists(path):
            return pd.read_parquet(path)

    elif expected_type == "numpy":
        # Check for npy extension
        path = base_path if base_path.endswith(".npy") else base_path + ".npy"
        if os.path.exists(path):
            return np.load(path)

    return None
