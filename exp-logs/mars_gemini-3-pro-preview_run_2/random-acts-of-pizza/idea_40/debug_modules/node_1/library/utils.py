import os
import sys
import random
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Configures a logger to output to both console and a file.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def suppress_warnings():
    """
    Suppresses warnings to keep the output clean.
    """
    warnings.filterwarnings("ignore")
    # Specific suppression for libraries that might be chatty
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensure_directory(path: str):
    """
    Ensures that the directory for a given file path exists.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_joblib(obj, path: str):
    """
    Saves a Python object (model, pipeline) using joblib.
    """
    ensure_directory(path)
    joblib.dump(obj, path)


def load_joblib(path: str):
    """
    Loads a Python object using joblib.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def process_with_cache(
    cache_path: str,
    process_fn,
    load_cached_data: bool = True,
    save_format: str = "npy",
    **kwargs,
):
    """
    Executes a processing function with caching logic.

    Args:
        cache_path: Path where the result should be saved/loaded.
        process_fn: The function to execute if cache is missing or ignored.
        load_cached_data: Whether to attempt loading from cache.
        save_format: Format to save/load ('npy' for numpy arrays, 'parquet' for DataFrames).
        **kwargs: Arguments to pass to process_fn.

    Returns:
        The processed data (loaded or computed).
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            if save_format == "npy":
                data = np.load(cache_path)
                # Validation: Check if length matches input texts if provided
                # Cite debug_lesson_14: Enforce Schema Validation When Loading Cached Data
                if "texts" in kwargs:
                    expected_len = len(kwargs["texts"])
                    if len(data) != expected_len:
                        raise ValueError(
                            f"Cache length mismatch: expected {expected_len}, got {len(data)}"
                        )
                return data
            elif save_format == "parquet":
                data = pd.read_parquet(cache_path)
                # Validation: Check if length matches input texts if provided
                # Cite debug_lesson_14: Enforce Schema Validation When Loading Cached Data
                if "texts" in kwargs:
                    expected_len = len(kwargs["texts"])
                    if len(data) != expected_len:
                        raise ValueError(
                            f"Cache length mismatch: expected {expected_len}, got {len(data)}"
                        )
                return data
            else:
                raise ValueError(f"Unsupported save_format: {save_format}")
        except Exception as e:
            # If load fails or validation fails, proceed to re-compute
            pass

    # 2. Compute data
    ensure_directory(cache_path)
    data = process_fn(**kwargs)

    # 3. Save to cache
    if save_format == "npy":
        np.save(cache_path, data)
    elif save_format == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=False)
        else:
            raise TypeError("Data must be a pandas DataFrame for parquet format.")
    else:
        raise ValueError(f"Unsupported save_format: {save_format}")

    return data
