import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from library.config import SEED


def set_seed(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str):
    """
    Configures and returns a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid double logging if root is configured
    logger.propagate = False

    return logger


def save_cache(data, filename: str, directory: str):
    """
    Saves data to the specified directory using allowed formats (.parquet or .npy).

    Args:
        data: The data object (pandas DataFrame or numpy ndarray).
        filename (str): The name of the file (extension optional).
        directory (str): The target directory.
    """
    os.makedirs(directory, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if not filename.endswith(".parquet"):
            filename += ".parquet"
        path = os.path.join(directory, filename)
        data.to_parquet(path, index=False)

    elif isinstance(data, np.ndarray):
        if not filename.endswith(".npy"):
            filename += ".npy"
        path = os.path.join(directory, filename)
        np.save(path, data)

    else:
        raise TypeError(
            f"Unsupported data type for caching: {type(data)}. Only DataFrame and ndarray are supported."
        )


def load_cache(filename: str, directory: str):
    """
    Attempts to load data from the cache directory.

    Args:
        filename (str): The name of the file (extension optional).
        directory (str): The source directory.

    Returns:
        The loaded data (DataFrame or ndarray) if found, else None.
    """
    # Normalize filename to check both extensions if not provided
    base_name = os.path.splitext(filename)[0]

    # Check for Parquet
    parquet_path = os.path.join(directory, base_name + ".parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    # Check for Numpy
    npy_path = os.path.join(directory, base_name + ".npy")
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)

    return None


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints validation metrics with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the output line.
    """
    prefix_str = f"{prefix} " if prefix else ""
    for key, value in metrics.items():
        print(f"{prefix_str}{key}: {value}")
