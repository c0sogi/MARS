import os
import sys
import random
import logging
import numpy as np
import pandas as pd
from library.config import SEED


def set_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: Torch is not imported here to keep dependencies minimal,
    # as the solution primarily uses sklearn/numpy.


def get_logger(name: str = "main") -> logging.Logger:
    """
    Configures and returns a logger that writes to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(file_path: str) -> None:
    """
    Ensures that the directory for a given file path exists.

    Args:
        file_path (str): The full path to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Saves a pandas DataFrame to a parquet file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The source file path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)


def save_npy(arr: np.ndarray, path: str) -> None:
    """
    Saves a numpy array to a .npy file, ensuring the directory exists.

    Args:
        arr (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    ensure_dir(path)
    np.save(path, arr)


def load_npy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The source file path.

    Returns:
        np.ndarray: The loaded array.
    """
    return np.load(path, allow_pickle=True)


def print_metrics(metrics: dict, prefix: str = "Validation") -> None:
    """
    Prints metric values with full precision (no rounding).

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Prefix string for the log output.
    """
    print(f"--- {prefix} Metrics ---")
    for key, value in metrics.items():
        # Using repr() or generic formatting to ensure full float precision is displayed
        print(f"{key}: {value}")
    print("-------------------------")
