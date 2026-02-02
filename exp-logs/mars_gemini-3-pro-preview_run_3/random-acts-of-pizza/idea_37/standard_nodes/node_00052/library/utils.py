import os
import time
import random
import contextlib
import numpy as np
import pandas as pd
import torch
from library import config


def set_seed(seed=config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def timer(name):
    """
    Context manager to measure and print the execution time of a block of code.
    """
    t0 = time.time()
    print(f"[{name}] Starting...")
    yield
    elapsed = time.time() - t0
    print(f"[{name}] Done in {elapsed:.3f} s")


def load_dataset(split):
    """
    Loads the specified dataset split (train, val, or test) from the metadata directory
    as defined in config.py.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = config.TRAIN_PATH
    elif split == "val":
        path = config.VAL_PATH
    elif split == "test":
        path = config.TEST_PATH
    else:
        raise ValueError(
            f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found for split '{split}' at {path}")

    return pd.read_parquet(path)


def save_cache(data, filename):
    """
    Saves data to the cache directory defined in config.py.
    Enforces the use of Parquet for DataFrames and NPY/NPZ for NumPy arrays.

    Args:
        data: The data object to save (pd.DataFrame, np.ndarray, or dict of arrays).
        filename (str): The name of the file (e.g., 'features.parquet').
    """
    filepath = os.path.join(config.CACHE_DIR, filename)

    # Ensure cache directory exists (redundant if config handles it, but safe)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if filename.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Parquet format is only supported for pandas DataFrames.")
    elif filename.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(filepath, data)
        else:
            raise ValueError("NPY format is only supported for numpy arrays.")
    elif filename.endswith(".npz"):
        if isinstance(data, dict):
            np.savez_compressed(filepath, **data)
        else:
            raise ValueError("NPZ format expects a dictionary of numpy arrays.")
    else:
        raise ValueError("Unsupported file format. Please use .parquet, .npy, or .npz.")


def load_cache(filename):
    """
    Attempts to load data from the cache directory.

    Args:
        filename (str): The name of the file to load.

    Returns:
        The loaded data object, or None if the file does not exist.
    """
    filepath = os.path.join(config.CACHE_DIR, filename)

    if not os.path.exists(filepath):
        return None

    if filename.endswith(".parquet"):
        return pd.read_parquet(filepath)
    elif filename.endswith(".npy"):
        return np.load(filepath, allow_pickle=True)
    elif filename.endswith(".npz"):
        # Return as a dict-like object (NpzFile)
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError("Unsupported file format in cache.")
