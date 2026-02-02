import os
import random
import numpy as np
import torch
import pandas as pd
from typing import Union, Dict, Any
from library.config import Config


def seed_everything(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
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


def ensure_directory(path: str):
    """
    Ensures that the directory for the given path exists.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    ensure_directory(path)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The file path to load from.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def save_numpy(data: Union[np.ndarray, Dict[str, np.ndarray]], path: str):
    """
    Saves a numpy array or dictionary of arrays to a file.
    Uses .npz if data is a dictionary or path ends in .npz, otherwise .npy.

    Args:
        data: The numpy array or dictionary of arrays.
        path (str): The destination file path.
    """
    ensure_directory(path)
    if isinstance(data, dict) or path.endswith(".npz"):
        if isinstance(data, dict):
            np.savez(path, **data)
        else:
            np.savez(path, data=data)
    else:
        np.save(path, data)


def load_numpy(path: str) -> Any:
    """
    Loads numpy data from a .npy or .npz file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded numpy array or NpzFile object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # allow_pickle=True is required for object arrays (e.g. strings) which are common in NLP tasks
    return np.load(path, allow_pickle=True)


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
