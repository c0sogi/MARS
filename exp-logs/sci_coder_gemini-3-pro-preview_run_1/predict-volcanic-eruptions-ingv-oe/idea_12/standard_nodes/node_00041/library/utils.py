import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae_score(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)


def save_pickle(obj, path):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj (object): The Python object to save.
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
        object: The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_npy(array, path):
    """
    Saves a numpy array to a .npy file.
    Used for strict caching of numerical data.

    Args:
        array (np.ndarray): The numpy array to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The source file path.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    return np.load(path)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    Used for strict caching of tabular features.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The source file path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)
