import os
import json
import pandas as pd
import numpy as np
from library.config import set_seed


def ensure_dir(file_path):
    """
    Ensures that the directory for the given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_data(data, file_path):
    """
    Saves data to the specified file path. Supports .parquet, .npy, and .json.

    Args:
        data: The data object to save.
        file_path (str): The destination path including extension.
    """
    ensure_dir(file_path)

    if file_path.endswith(".parquet"):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame to save as .parquet")
        data.to_parquet(file_path, index=False)

    elif file_path.endswith(".npy"):
        # Convert to numpy array if not already, to ensure safe saving
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np.save(file_path, data)

    elif file_path.endswith(".json"):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    else:
        raise ValueError(f"Unsupported file extension for saving: {file_path}")


def load_data(file_path):
    """
    Loads data from the specified file path. Supports .parquet, .npy, and .json.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        The loaded data object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    elif file_path.endswith(".npy"):
        # allow_pickle=True is set to allow loading object arrays (e.g. strings),
        # but complex object serialization should use Parquet/JSON where possible.
        return np.load(file_path, allow_pickle=True)

    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file extension for loading: {file_path}")


def get_or_compute(file_path, compute_func, load_cached_data=True, **kwargs):
    """
    Retrieves data from cache if available and requested; otherwise computes it,
    saves it to the cache, and returns it.

    Args:
        file_path (str): Path to the cache file.
        compute_func (callable): Function to compute the data if cache is missed.
        load_cached_data (bool): If True, attempts to load from file_path first.
        **kwargs: Arguments passed to compute_func.

    Returns:
        The data (either loaded from cache or newly computed).
    """
    if load_cached_data and os.path.exists(file_path):
        try:
            return load_data(file_path)
        except Exception:
            # If loading fails, fall through to recompute
            pass

    # Compute data
    data = compute_func(**kwargs)

    # Save data
    save_data(data, file_path)

    return data
