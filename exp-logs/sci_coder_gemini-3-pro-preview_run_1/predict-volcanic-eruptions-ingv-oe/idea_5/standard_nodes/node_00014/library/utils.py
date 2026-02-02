import os
import random
import json
import hashlib
import numpy as np
import torch
import pandas as pd
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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash(config_dict):
    """
    Generates a unique MD5 hash string based on the provided configuration dictionary.
    This is used to create unique filenames for cached data based on parameters.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The MD5 hash string.
    """
    # Sort keys to ensure consistent ordering for hashing regardless of dict insertion order
    s = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def save_to_parquet(df, path):
    """
    Saves a pandas DataFrame to a parquet file, ensuring the parent directory exists.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_from_parquet(path):
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The file path to load.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if the file does not exist.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_to_npy(array, path):
    """
    Saves a numpy array to an npy file, ensuring the parent directory exists.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_from_npy(path):
    """
    Loads a numpy array from an npy file.

    Args:
        path (str): The file path to load.

    Returns:
        np.ndarray or None: The loaded array, or None if the file does not exist.
    """
    if os.path.exists(path):
        return np.load(path)
    return None
