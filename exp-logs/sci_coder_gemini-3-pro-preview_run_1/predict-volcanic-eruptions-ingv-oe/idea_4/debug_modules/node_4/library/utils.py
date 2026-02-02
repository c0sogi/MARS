import os
import random
import hashlib
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across random, numpy, and torch libraries.

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


def get_config_hash(config_dict):
    """
    Generates an MD5 hash for a dictionary configuration.
    Useful for creating unique cache keys based on hyperparameters.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Sort keys to ensure consistent ordering regardless of insertion order
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def save_pickle(obj, path):
    """
    Saves an object to a pickle file, ensuring the directory exists.

    Args:
        obj (object): The Python object to save.
        path (str): The file path where the object should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads an object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        object: The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred)
