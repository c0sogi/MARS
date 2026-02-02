import os
import sys
import logging
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def setup_logger(name: str = "pipeline", log_file: str = None, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, defaults to a file in WORKING_DIR.
        level: Logging level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file is None:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "execution.log")

    # Ensure directory exists for log file
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_fingerprint(obj) -> str:
    """
    Generates a stable MD5 hash for a given object (dict, list, string, etc.).
    Used for parameter-aware caching.

    Args:
        obj: The object to hash.

    Returns:
        Hex digest string of the hash.
    """
    try:
        # specific handling for dictionaries to ensure key order doesn't affect hash
        if isinstance(obj, dict):
            encoded = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        elif isinstance(obj, (list, tuple)):
            encoded = json.dumps(obj, default=str).encode("utf-8")
        else:
            encoded = str(obj).encode("utf-8")
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        encoded = str(obj).encode("utf-8")

    return hashlib.md5(encoded).hexdigest()


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df: The dataframe to optimize.
        verbose: Whether to print memory reduction statistics.

    Returns:
        Optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
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
    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def seed_everything(seed: int = Config.SEED):
    """
    Sets seeds for all random number generators to ensure reproducibility.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
