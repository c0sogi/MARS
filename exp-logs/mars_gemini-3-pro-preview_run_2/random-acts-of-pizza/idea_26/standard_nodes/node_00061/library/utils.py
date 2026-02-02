import os
import random
import numpy as np
import torch
import joblib
import pandas as pd
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directory(path: str) -> None:
    """
    Ensures that the directory for the given path exists.

    Args:
        path (str): The file path or directory path.
    """
    # If path has an extension, assume it's a file and get the directory
    if os.path.splitext(path)[1]:
        dirname = os.path.dirname(path)
    else:
        dirname = path

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_joblib(obj, path: str) -> None:
    """
    Saves a Python object using joblib. Useful for models and pipelines.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    ensure_directory(path)
    joblib.dump(obj, path)


def load_joblib(path: str):
    """
    Loads a Python object using joblib.

    Args:
        path (str): The source file path.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


def save_npy(array: np.ndarray, path: str) -> None:
    """
    Saves a numpy array to a .npy file. Useful for embeddings.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    ensure_directory(path)
    np.save(path, array)


def load_npy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The source file path.

    Returns:
        np.ndarray: The loaded array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Saves a pandas DataFrame to a parquet file. Useful for tabular features.

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
        path (str): The source file path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)
