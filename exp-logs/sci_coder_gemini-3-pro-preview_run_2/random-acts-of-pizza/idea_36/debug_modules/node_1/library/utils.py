import os
import random
import logging
import numpy as np
import torch
import joblib
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Configures a logger to output to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if the logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_pickle(obj, path):
    """
    Saves a Python object (e.g., model, pipeline) using joblib.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    """
    Loads a Python object using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    return joblib.load(path)


def save_npy(array, path):
    """
    Saves a numpy array to a .npy file.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The file path to load from.

    Returns:
        np.ndarray: The loaded array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a .parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a .parquet file.

    Args:
        path (str): The file path to load from.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)
