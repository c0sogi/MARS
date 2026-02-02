import os
import random
import numpy as np
import pandas as pd
import torch
import tensorflow as tf


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    # Python's built-in random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # OS environment for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch might not be installed or GPU might not be available
        pass

    # TensorFlow
    try:
        tf.random.set_seed(seed)
        # Prevent TF from using full GPU memory by default
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except Exception:
        # TF might not be installed
        pass


def print_header(title: str):
    """
    Prints a formatted header to the console.

    Args:
        title (str): The title text to display.
    """
    print(f"\n{'='*10} {title} {'='*10}")


def print_metric(name: str, value: float):
    """
    Prints a metric with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")


def ensure_dir(path: str):
    """
    Ensures that the directory for a given file path exists.

    Args:
        path (str): The file path or directory path.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination path.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a Parquet file.

    Args:
        path (str): The source path.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_parquet(path)


def save_numpy(data: np.ndarray, path: str):
    """
    Saves a numpy array to a .npy file.

    Args:
        data (np.ndarray): The array to save.
        path (str): The destination path.
    """
    ensure_dir(path)
    np.save(path, data)


def load_numpy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The source path.

    Returns:
        np.ndarray: The loaded array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    return np.load(path)


def save_numpy_compressed(data_dict: dict, path: str):
    """
    Saves multiple numpy arrays to a compressed .npz file.

    Args:
        data_dict (dict): Dictionary of arrays to save.
        path (str): The destination path.
    """
    ensure_dir(path)
    np.savez_compressed(path, **data_dict)


def load_numpy_compressed(path: str):
    """
    Loads a compressed .npz file.

    Args:
        path (str): The source path.

    Returns:
        NpzFile: The loaded numpy archive.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    return np.load(path)
