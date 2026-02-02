import os
import sys
import random
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(name="pipeline", level=logging.INFO):
    """
    Configures and returns a logger instance that writes to stdout.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_artifact(obj, path):
    """
    Saves a Python object (e.g., model, transformer) using joblib.
    Creates the parent directory if it does not exist.

    Args:
        obj: The object to save.
        path (str): The file path to save to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path):
    """
    Loads a Python object using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def save_data_cache(data, path):
    """
    Saves data to disk using Parquet (for DataFrames) or NumPy (for arrays).
    Strictly avoids pickle for data storage to comply with requirements.

    Args:
        data: pandas.DataFrame or numpy.ndarray.
        path (str): The base file path. Extension may be appended automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if not path.endswith(".parquet"):
            path += ".parquet"
        data.to_parquet(path, index=False)
    elif isinstance(data, np.ndarray):
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)
    else:
        raise ValueError(
            "save_data_cache only supports pandas DataFrame (parquet) or numpy array (npy)."
        )


def load_data_cache(path):
    """
    Loads data from disk (Parquet or NumPy).

    Args:
        path (str): The file path.

    Returns:
        pandas.DataFrame or numpy.ndarray.
    """
    # Handle cases where extension might be missing in the call but present on disk
    if not os.path.exists(path):
        if os.path.exists(path + ".parquet"):
            path += ".parquet"
        elif os.path.exists(path + ".npy"):
            path += ".npy"
        else:
            raise FileNotFoundError(f"Cache file not found at {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for cached data: {path}")
