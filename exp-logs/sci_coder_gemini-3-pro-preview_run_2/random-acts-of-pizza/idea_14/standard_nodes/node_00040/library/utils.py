import os
import sys
import random
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_object(obj, file_path):
    """
    Saves a Python object (e.g., model, scaler) using joblib.

    Args:
        obj: The object to save.
        file_path (str): Destination path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Loads a Python object using joblib.

    Args:
        file_path (str): Path to the file.

    Returns:
        The loaded object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return joblib.load(file_path)


def save_data(df, file_path):
    """
    Saves a pandas DataFrame to parquet.
    Used for caching deterministic data processing, avoiding pickle for data.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): Destination path.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path, index=False)


def load_data(file_path):
    """
    Loads a pandas DataFrame from parquet.

    Args:
        file_path (str): Path to the file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_parquet(file_path)


def validate_data_integrity(df, name="Data", expected_rows=None):
    """
    Validates the integrity of a DataFrame.
    Prints shape and checks against expected rows if provided.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        name (str): Name of the dataset for logging.
        expected_rows (int, optional): Expected number of rows.
    """
    print(f"[{name}] Shape: {df.shape}")

    if expected_rows is not None:
        if len(df) != expected_rows:
            print(f"[{name}] WARNING: Expected {expected_rows} rows, got {len(df)}")
        else:
            print(f"[{name}] Row count verification passed: {len(df)}")

    # Check for nulls in columns
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[{name}] Null values detected:")
        print(null_counts[null_counts > 0])
    else:
        print(f"[{name}] No null values found.")
