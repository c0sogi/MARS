import os
import random
import numpy as np
import torch
import joblib
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directory(path: str):
    """
    Ensures that the directory for the given path exists.

    Args:
        path (str): The file path for which the directory should exist.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_joblib(obj, path: str):
    """
    Saves a Python object using joblib.
    Useful for saving models and pipelines.

    Args:
        obj: The Python object to save.
        path (str): The destination file path.
    """
    ensure_directory(path)
    joblib.dump(obj, path)


def load_joblib(path: str):
    """
    Loads a Python object using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded Python object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


def save_numpy(array: np.ndarray, path: str):
    """
    Saves a numpy array to a .npy file.

    Args:
        array (np.ndarray): The numpy array to save.
        path (str): The destination file path.
    """
    ensure_directory(path)
    np.save(path, array)


def load_numpy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The file path to load from.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def print_metric(name: str, value: float):
    """
    Prints a metric with full precision without rounding.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
