import os
import random
import numpy as np
import torch
import pickle
import hashlib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def generate_file_hash(filepath):
    """
    Generates a unique hash for a file based on its modification time and size.
    Used for caching mechanisms to detect if source data has changed.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: MD5 hash of the file's metadata (size + mtime), or None if file does not exist.
    """
    if not os.path.exists(filepath):
        return None

    # Get file stats
    stats = os.stat(filepath)
    # Create a unique identifier string based on size and mtime
    identifier = f"{stats.st_size}_{stats.st_mtime}"
    # Return MD5 hash of the identifier
    return hashlib.md5(identifier.encode("utf-8")).hexdigest()


def save_pickle(obj, path):
    """
    Saves a Python object to a pickle file.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The source file path.

    Returns:
        The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df, path):
    """
    Saves a Pandas DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a Pandas DataFrame from a parquet file.

    Args:
        path (str): The source file path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)


def save_npy(array, path):
    """
    Saves a NumPy array to a .npy file.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path):
    """
    Loads a NumPy array from a .npy file.

    Args:
        path (str): The source file path.

    Returns:
        np.ndarray: The loaded array.
    """
    return np.load(path)
