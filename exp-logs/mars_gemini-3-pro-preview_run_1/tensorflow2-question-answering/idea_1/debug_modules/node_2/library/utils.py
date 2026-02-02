import os
import sys
import time
import random
import logging
import contextlib
import numpy as np
import pandas as pd
import torch
from typing import Callable, Any, Optional


def setup_logger(name: str = "main", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger with a standard format.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across Python, Numpy, and Torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


@contextlib.contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to measure and log the execution time of a block of code.

    Args:
        name (str): Description of the code block being timed.
        logger (logging.Logger, optional): Logger to use for output. If None, uses print.
    """
    t0 = time.time()
    msg_start = f"[{name}] Start"

    if logger:
        logger.info(msg_start)
    else:
        print(msg_start)

    yield

    elapsed = time.time() - t0
    msg_end = f"[{name}] Done in {elapsed:.2f} s"

    if logger:
        logger.info(msg_end)
    else:
        print(msg_end)


def load_or_create_cache(
    file_path: str,
    process_fn: Callable[[], Any],
    load_cached_data: bool,
    file_type: str = "parquet",
) -> Any:
    """
    Generic caching mechanism for deterministic data processing.

    Logic:
    1. IF load_cached_data is True: Try to load the file.
    2. IF loading fails OR load_cached_data is False:
       - Compute data using process_fn().
       - Save result to file_path.

    Args:
        file_path (str): Full path to the cache file.
        process_fn (Callable): Function to generate data if cache is missed.
        load_cached_data (bool): Configuration flag to enable/disable loading.
        file_type (str): Format to save/load ('parquet' for DataFrame, 'npy' for Numpy).

    Returns:
        The loaded or computed data.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = None
    loaded = False

    # 1. Try to load
    if load_cached_data and os.path.exists(file_path):
        try:
            if file_type == "parquet":
                data = pd.read_parquet(file_path)
            elif file_type == "npy":
                data = np.load(file_path, allow_pickle=False)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")

            loaded = True
            # Optional: Log success if a logger was available, but we keep it silent/standard here.
        except Exception:
            # Fallback to processing if load fails
            loaded = False

    # 2. Process if not loaded
    if not loaded:
        data = process_fn()

        # Save result
        if file_type == "parquet":
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, index=False)
            else:
                raise TypeError("Data must be a pandas DataFrame for parquet caching.")
        elif file_type == "npy":
            if isinstance(data, np.ndarray):
                np.save(file_path, data)
            else:
                raise TypeError("Data must be a numpy ndarray for npy caching.")

    return data
