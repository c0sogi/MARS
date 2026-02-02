import os
import random
import numpy as np
import torch
import pandas as pd
import logging
import time
from contextlib import contextmanager
from library.config import SEED, METADATA_DIR, CACHE_DIR


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Random seed set to {seed}")


def get_logger(name):
    """
    Returns a configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_dataset(split, nrows=None):
    """
    Loads the dataset for a specific split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.
        nrows (int, optional): Number of rows to load (for debugging).

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

    file_path = os.path.join(METADATA_DIR, f"{split}.parquet")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    # Pandas read_parquet does not support nrows directly in all versions/engines efficiently,
    # but we can load and slice if necessary, or use specific engine features.
    # For simplicity and compatibility, we load then slice.
    df = pd.read_parquet(file_path)

    if nrows is not None:
        df = df.head(nrows)

    return df


def save_to_cache(data, filename, format="parquet"):
    """
    Saves data to the defined CACHE_DIR.

    Args:
        data: The object to save (DataFrame or Numpy array).
        filename (str): The name of the file.
        format (str): 'parquet', 'npy', or 'npz'.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_path = os.path.join(CACHE_DIR, filename)

    if format == "parquet":
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame for parquet format.")
        data.to_parquet(file_path, index=False)
    elif format == "npy":
        np.save(file_path, data)
    elif format == "npz":
        if isinstance(data, dict):
            np.savez(file_path, **data)
        else:
            # Assume data is a list/tuple of arrays or a single array
            np.savez(file_path, data)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # print(f"Saved {filename} to cache.")


def load_from_cache(filename, format="parquet"):
    """
    Loads data from the defined CACHE_DIR.

    Args:
        filename (str): The name of the file.
        format (str): 'parquet', 'npy', or 'npz'.

    Returns:
        The loaded data, or None if file does not exist.
    """
    file_path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(file_path):
        return None

    if format == "parquet":
        return pd.read_parquet(file_path)
    elif format == "npy":
        return np.load(file_path, allow_pickle=True)
    elif format == "npz":
        return np.load(file_path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def print_metrics(metrics_dict, title="Metrics"):
    """
    Prints metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        title (str): Title for the print block.
    """
    print(f"--- {title} ---")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
    print("----------------")


@contextmanager
def timer(name):
    """
    Context manager to measure execution time.
    """
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] done in {t1 - t0} s")
