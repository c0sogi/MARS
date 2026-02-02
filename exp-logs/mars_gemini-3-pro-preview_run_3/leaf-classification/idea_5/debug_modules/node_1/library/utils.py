import os
import random
import numpy as np
import torch
import joblib
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, Numpy, and PyTorch.

    Args:
        seed (int): The random seed to set. Defaults to Config.SEED.
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


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Clips probabilities to the range [1e-15, 1-1e-15] as per the metric definition
    to avoid extremes of the log function.

    Args:
        probs (np.ndarray): The array of probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    eps = 1e-15
    return np.clip(probs, eps, 1.0 - eps)


def save_numpy(data: np.ndarray, path: str) -> None:
    """
    Saves a numpy array to the specified path, creating parent directories if needed.

    Args:
        data (np.ndarray): The data to save.
        path (str): The file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_numpy(path: str):
    """
    Loads a numpy array from the specified path.

    Args:
        path (str): The file path.

    Returns:
        np.ndarray or None: The loaded data, or None if the file is missing or corrupt.
    """
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def save_pickle(obj, path: str) -> None:
    """
    Saves a Python object (e.g., model pipeline) to the specified path using joblib.

    Args:
        obj: The object to save.
        path (str): The file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: str):
    """
    Loads a Python object from the specified path using joblib.

    Args:
        path (str): The file path.

    Returns:
        object or None: The loaded object, or None if the file is missing or corrupt.
    """
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
