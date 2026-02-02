import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED, CACHE_DIR


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_path(filename):
    """
    Resolves the full path for a file within the configured CACHE_DIR.
    """
    return os.path.join(CACHE_DIR, filename)


def save_to_cache(data, filename):
    """
    Saves data to the cache directory using appropriate formats.
    - pandas.DataFrame -> .parquet
    - numpy.ndarray -> .npy
    - dict (of arrays) -> .npz
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    filepath = get_cache_path(filename)

    if isinstance(data, pd.DataFrame):
        if not filepath.endswith(".parquet"):
            filepath += ".parquet"
        data.to_parquet(filepath, index=False)

    elif isinstance(data, np.ndarray):
        if not filepath.endswith(".npy"):
            filepath += ".npy"
        np.save(filepath, data)

    elif isinstance(data, dict):
        # Assumes dictionary of numpy arrays for npz
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        np.savez(filepath, **data)
    else:
        raise ValueError(
            f"Unsupported data type for caching: {type(data)}. Use DataFrame, ndarray, or dict of arrays."
        )

    # print(f"Cached data saved to: {filepath}")


def load_from_cache(filename):
    """
    Attempts to load data from the cache directory.
    Returns the loaded data if found, else None.
    Handles extension inference if not provided in filename.
    """
    filepath = get_cache_path(filename)

    # Attempt to find the file with supported extensions if not explicitly provided
    found_path = None
    if os.path.exists(filepath):
        found_path = filepath
    else:
        for ext in [".parquet", ".npy", ".npz"]:
            if os.path.exists(filepath + ext):
                found_path = filepath + ext
                break

    if found_path is None:
        return None

    if found_path.endswith(".parquet"):
        return pd.read_parquet(found_path)
    elif found_path.endswith(".npy"):
        return np.load(found_path)
    elif found_path.endswith(".npz"):
        return np.load(found_path)
    else:
        return None


def print_metrics(metrics_dict, prefix=""):
    """
    Prints validation metrics with full precision (no rounding).
    """
    header = f"--- {prefix} Metrics ---" if prefix else "--- Metrics ---"
    print(header)
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
