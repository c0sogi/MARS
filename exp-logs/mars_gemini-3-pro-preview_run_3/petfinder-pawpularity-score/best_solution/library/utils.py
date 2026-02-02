import os
import random
import numpy as np
import torch
import joblib


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_pickle(obj, path: str):
    """
    Saves a Python object (e.g., a model) to a file using joblib.
    Ensures the directory exists before saving.

    Args:
        obj: The object to save.
        path (str): The file path where the object should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: str):
    """
    Loads a Python object from a file using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    return joblib.load(path)


def save_numpy(array: np.ndarray, path: str):
    """
    Saves a numpy array to a .npy file.
    Ensures the directory exists before saving.

    Args:
        array (np.ndarray): The array to save.
        path (str): The file path where the array should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_numpy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The file path to load from.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    return np.load(path)
