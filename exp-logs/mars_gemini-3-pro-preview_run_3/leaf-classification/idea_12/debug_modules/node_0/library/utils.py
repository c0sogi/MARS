import os
import random
import pickle
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_pickle(data, path):
    """
    Saves a Python object to a file using pickle.
    Useful for saving model artifacts like sklearn pipelines.

    Args:
        data: The Python object to save.
        path (str): The file path where the object should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    Preferred format for caching tabular data.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The file path where the DataFrame should be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The file path to load from.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)


def save_npy(array, path):
    """
    Saves a numpy array to a .npy file.
    Preferred format for caching feature arrays.

    Args:
        array (np.ndarray): The array to save.
        path (str): The file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The file path.

    Returns:
        np.ndarray: The loaded array.
    """
    return np.load(path)
