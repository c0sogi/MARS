import os
import random
import numpy as np
import torch
import pandas as pd
import joblib
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def load_data(path, debug=False, n_rows=None):
    """
    Loads data from a Parquet file.

    Args:
        path (str): Path to the parquet file.
        debug (bool): If True, loads a subset of data.
        n_rows (int): Number of rows to load if debug is True. Default is 100.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if debug:
        # Load a small subset for debugging
        limit = n_rows if n_rows is not None else 100
        # Read full then head is safer for parquet if nrows not supported by engine version,
        # though pyarrow usually supports reading row groups. For this dataset size, full read is fine.
        df = pd.read_parquet(path)
        df = df.head(limit)
    else:
        df = pd.read_parquet(path)

    return df


def save_cache(data, filename, directory=Config.WORKING_DIR):
    """
    Saves data to cache using .npy for numpy arrays or .parquet for DataFrames.
    Strictly avoids pickle for data as per requirements.

    Args:
        data: The data object (np.ndarray or pd.DataFrame).
        filename (str): The name of the file (extension will be appended/corrected).
        directory (str): The directory to save to.
    """
    os.makedirs(directory, exist_ok=True)

    # Remove extension if provided to ensure correct one is added
    base_name = os.path.splitext(filename)[0]

    if isinstance(data, pd.DataFrame):
        path = os.path.join(directory, f"{base_name}.parquet")
        data.to_parquet(path, index=False)
    elif isinstance(data, np.ndarray):
        path = os.path.join(directory, f"{base_name}.npy")
        np.save(path, data)
    elif isinstance(data, dict):
        # If dictionary contains only arrays, save as npz
        path = os.path.join(directory, f"{base_name}.npz")
        np.savez(path, **data)
    else:
        raise ValueError(
            "save_cache only supports pd.DataFrame (parquet), np.ndarray (npy), or dict of arrays (npz)."
        )


def load_cache(filename, directory=Config.WORKING_DIR):
    """
    Loads data from cache (npy, npz, or parquet).

    Args:
        filename (str): The name of the file (without extension).
        directory (str): The directory to load from.

    Returns:
        The loaded data or None if not found.
    """
    base_name = os.path.splitext(filename)[0]

    parquet_path = os.path.join(directory, f"{base_name}.parquet")
    npy_path = os.path.join(directory, f"{base_name}.npy")
    npz_path = os.path.join(directory, f"{base_name}.npz")

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(npy_path):
        return np.load(npy_path)
    elif os.path.exists(npz_path):
        return np.load(npz_path)

    return None


def get_cached_data(
    func, cache_name, load_cached_data=True, directory=Config.WORKING_DIR, **kwargs
):
    """
    Generic caching wrapper implementing the required logic flow.

    Args:
        func (callable): The function to compute data if cache misses.
        cache_name (str): The name of the cache file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.
        directory (str): Cache directory.
        **kwargs: Arguments to pass to func.

    Returns:
        The data (loaded or computed).
    """
    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data:
        data = load_cache(cache_name, directory)
        if data is not None:
            return data

    # 2. IF loading fails OR load_cached_data is False: Compute and Save.
    data = func(**kwargs)
    save_cache(data, cache_name, directory)

    return data


def save_pickle(obj, filename, directory=Config.WORKING_DIR):
    """
    Helper to save generic Python objects (like models) using joblib.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(obj, path)


def load_pickle(filename, directory=Config.WORKING_DIR):
    """
    Helper to load generic Python objects using joblib.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def save_predictions(ids, probabilities, output_path=Config.SUBMISSION_FILE):
    """
    Saves predictions to the submission file format.

    Args:
        ids (array-like): Request IDs.
        probabilities (array-like): Predicted probabilities for class 1.
        output_path (str): Path to save CSV.
    """
    # Ensure inputs are 1D arrays
    ids = np.array(ids).flatten()
    probabilities = np.array(probabilities).flatten()

    df = pd.DataFrame({"request_id": ids, "requester_received_pizza": probabilities})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
