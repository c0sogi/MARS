import os
import json
import hashlib
import numpy as np
import pandas as pd
from library.config import Config, set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in library.config.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def generate_cache_key(config_dict):
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    Used to version cache files based on hyperparameters.

    Args:
        config_dict (dict): Dictionary containing configuration parameters
                            (e.g., window_size, feature_list).

    Returns:
        str: The first 8 characters of the MD5 hash of the sorted JSON representation.
    """
    if config_dict is None:
        return "default"

    # Serialize dictionary to JSON with sorted keys to ensure determinism
    # default=str handles non-serializable types by converting them to strings
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Compute MD5 hash
    hash_obj = hashlib.md5(config_str.encode("utf-8"))

    # Return first 8 characters for a concise suffix
    return hash_obj.hexdigest()[:8]


def get_hashed_path(base_name, config_dict, ext=".parquet"):
    """
    Constructs a full file path including the working directory and a hash suffix.

    Args:
        base_name (str): The base name of the file (e.g., 'train_features').
        config_dict (dict): Configuration dictionary to generate the hash.
        ext (str): File extension (default: .parquet).

    Returns:
        str: Full path to the file (e.g., ./working/idea_3/train_features_a1b2c3d4.parquet).
    """
    key = generate_cache_key(config_dict)
    filename = f"{base_name}_{key}{ext}"
    return os.path.join(Config.WORKING_DIR, filename)


def save_cached_data(data, path):
    """
    Saves data to the specified path, ensuring the directory exists.
    Supports pandas DataFrames (parquet) and numpy arrays (npy).

    Args:
        data: The data object to save (pd.DataFrame or np.ndarray).
        path (str): The destination file path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
    elif isinstance(data, np.ndarray):
        np.save(path, data)
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. Use pd.DataFrame or np.ndarray."
        )


def load_cached_data(path):
    """
    Loads data from the specified path if it exists.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded data (pd.DataFrame or np.ndarray) or None if file does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for path: {path}")


def print_metric(name, value):
    """
    Prints a metric with full precision.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
