import os
import random
import numpy as np
import torch
import pandas as pd
import joblib
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(file_path):
    """
    Ensures the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_cache_path(filename):
    """
    Resolves a filename to the full path within the Config.CACHE_DIR.
    """
    return os.path.join(Config.CACHE_DIR, filename)


def save_parquet(df, filename):
    """
    Saves a pandas DataFrame to a parquet file in the cache directory.
    """
    path = get_cache_path(filename)
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(filename):
    """
    Loads a pandas DataFrame from a parquet file in the cache directory.
    Returns None if file does not exist.
    """
    path = get_cache_path(filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_npy(data, filename):
    """
    Saves a dictionary of numpy arrays to a compressed .npz file in the cache directory.
    Usage: save_npy({'X': x_arr, 'y': y_arr}, 'data.npz')
    """
    path = get_cache_path(filename)
    ensure_dir(path)
    np.savez_compressed(path, **data)


def load_npy(filename):
    """
    Loads a .npz file from the cache directory.
    Returns a dictionary-like object (NpzFile) or None if not found.
    """
    path = get_cache_path(filename)
    if os.path.exists(path):
        return np.load(path)
    return None


def save_pickle(obj, filename):
    """
    Saves a generic Python object (e.g., sklearn model) using joblib to the cache directory.
    """
    path = get_cache_path(filename)
    ensure_dir(path)
    joblib.dump(obj, path)


def load_pickle(filename):
    """
    Loads a generic Python object using joblib from the cache directory.
    Returns None if file does not exist.
    """
    path = get_cache_path(filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def save_torch_model(model, filename):
    """
    Saves a PyTorch model state dict to the cache directory.
    """
    path = get_cache_path(filename)
    ensure_dir(path)
    torch.save(model.state_dict(), path)


def load_torch_model(model, filename, device=Config.DEVICE):
    """
    Loads a PyTorch model state dict into the provided model instance from the cache directory.
    Returns True if successful, False if file not found.
    """
    path = get_cache_path(filename)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    return False


def save_submission(df):
    """
    Saves the submission DataFrame to the path specified in Config.
    """
    path = Config.SUBMISSION_PATH
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
