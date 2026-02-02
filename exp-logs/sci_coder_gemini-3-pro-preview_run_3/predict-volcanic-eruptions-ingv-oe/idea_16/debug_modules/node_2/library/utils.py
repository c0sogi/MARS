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
        seed (int): The seed value to use.
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


def save_artifact(obj, path):
    """
    Saves an object to the specified path.
    Uses Parquet for pandas DataFrames if the path ends in .parquet.
    Uses Numpy save for .npy files.
    Uses Pickle for other objects or paths ending in .pkl.

    Args:
        obj: The object to save (DataFrame, model, etc.).
        path (str): The destination file path.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if path.endswith(".parquet"):
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path, index=False)
        else:
            raise ValueError(
                f"Cannot save object of type {type(obj)} as parquet. Expected pd.DataFrame."
            )
    elif path.endswith(".npy"):
        np.save(path, obj)
    else:
        # Default to pickle for models and generic objects
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def load_artifact(path):
    """
    Loads an object from the specified path.
    Supports .parquet for DataFrames, .npy for numpy arrays, and .pkl for generic objects.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        # Default to pickle
        with open(path, "rb") as f:
            return pickle.load(f)
