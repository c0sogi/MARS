import os
import time
import random
import numpy as np
import pandas as pd
import joblib
import torch
from contextlib import contextmanager
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextmanager
def Timer(name="Task"):
    """
    Context manager to measure and print the execution time of a code block.
    Prints full precision time.
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[{name}] completed in {elapsed} seconds.")


def ensure_dir(file_path):
    """
    Ensures that the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a Parquet file.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a Parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def save_npy(arr, path):
    """
    Saves a numpy array to a .npy file.
    """
    ensure_dir(path)
    np.save(path, arr)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)


def save_joblib(obj, path):
    """
    Saves a Python object (e.g., model, transformer) using joblib.
    """
    ensure_dir(path)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads a Python object using joblib.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Joblib file not found: {path}")
    return joblib.load(path)
