import os
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
from library import config


def set_seed(seed=config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python's random, numpy,
    and environment variables.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Logger:
    """
    Simple logger to track pipeline execution status with timestamps.
    """

    @staticmethod
    def info(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [INFO] {message}")

    @staticmethod
    def error(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [ERROR] {message}")

    @staticmethod
    def metric(name, value):
        """
        Prints a metric with full precision (using repr) to avoid rounding.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [METRIC] {name}: {value!r}")


class Timer:
    """
    Context manager to measure execution time of a block.
    """

    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start_time = time.time()
        Logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        Logger.info(f"Finished: {self.description} (took {elapsed:.4f} seconds)")


def load_metadata(split):
    """
    Loads the metadata CSV for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = config.TRAIN_METADATA_PATH
    elif split == "val":
        path = config.VAL_METADATA_PATH
    elif split == "test":
        path = config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def save_cache_npy(path, array):
    """
    Saves a numpy array to the specified path, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
    Logger.info(f"Cached numpy array to {path}")


def load_cache_npy(path):
    """
    Loads a numpy array from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    Logger.info(f"Loaded cached numpy array from {path}")
    return np.load(path, allow_pickle=True)


def save_cache_parquet(path, df):
    """
    Saves a pandas DataFrame to parquet, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    Logger.info(f"Cached DataFrame to {path}")


def load_cache_parquet(path):
    """
    Loads a pandas DataFrame from parquet.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    Logger.info(f"Loaded cached DataFrame from {path}")
    return pd.read_parquet(path)
