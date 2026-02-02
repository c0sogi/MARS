import os
import random
import functools
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from scipy.ndimage import gaussian_filter1d
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Essential for deterministic behavior in the Anchored-Mining curriculum.
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


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for safety
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return matthews_corrcoef(y_true, y_pred)


def gaussian_smooth_labels(labels, sigma=1.0):
    """
    Applies Gaussian smoothing to a temporal sequence of binary labels.
    Used for Temporal Label Smoothing to address +/- 10Hz label noise.

    Args:
        labels (array-like): 1D sequence of binary contact labels.
        sigma (float): Standard deviation for Gaussian kernel (in timesteps).

    Returns:
        np.array: Smoothed continuous probability labels.
    """
    # Convert to float to allow continuous values
    labels_float = np.asarray(labels, dtype=float)
    return gaussian_filter1d(labels_float, sigma=sigma)


def load_metadata(path, debug=False, sample_size=5000):
    """
    Loads metadata CSV and optionally samples it for debugging/development.
    Preserves temporal integrity by sampling unique plays instead of random rows.

    Args:
        path (str): Path to the metadata CSV file.
        debug (bool): If True, performs sampling.
        sample_size (int): Number of unique plays to sample if debug is True.

    Returns:
        pd.DataFrame: The loaded (and potentially sampled) metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    if debug:
        # Sample by game_play to keep full temporal sequences for each play
        if "game_play" in df.columns:
            unique_plays = df["game_play"].unique()
            if len(unique_plays) > sample_size:
                rng = np.random.RandomState(Config.SEED)
                sampled_plays = rng.choice(
                    unique_plays, size=sample_size, replace=False
                )
                df = df[df["game_play"].isin(sampled_plays)].copy()
                print(
                    f"DEBUG: Sampled {len(sampled_plays)} plays ({len(df)} rows) from {path}"
                )
        else:
            # Fallback if game_play column is missing
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=Config.SEED).copy()
                print(f"DEBUG: Sampled {len(df)} rows from {path}")

    return df


def save_cache(data, path):
    """
    Saves data to the specified path, handling directory creation and format selection.

    Args:
        data: The data to save (pd.DataFrame or np.ndarray).
        path (str): The destination file path (.parquet or .npy).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            # Attempt to convert to DataFrame
            pd.DataFrame(data).to_parquet(path, index=False)
    elif path.endswith(".npy"):
        np.save(path, data)
    else:
        raise ValueError(f"Unsupported cache format: {path}. Use .parquet or .npy")


def load_cache(path):
    """
    Loads data from the specified path.

    Args:
        path (str): The file path to load.

    Returns:
        The loaded data (pd.DataFrame or np.ndarray), or None if path does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported cache format: {path}. Use .parquet or .npy")


def cache_processor(func):
    """
    Decorator for deterministic data processing functions.
    Implements the Check -> Load -> Compute -> Save workflow.

    The decorated function must accept 'load_cached_data' (bool) and 'cache_path' (str)
    in its keyword arguments (kwargs).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        load_cached = kwargs.get("load_cached_data", False)
        cache_path = kwargs.get("cache_path", None)

        # 1. Try to load from cache
        if load_cached and cache_path and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                data = load_cache(cache_path)
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}). Recomputing...")

        # 2. Compute
        result = func(*args, **kwargs)

        # 3. Save to cache
        if cache_path:
            print(f"Saving computed data to {cache_path}...")
            save_cache(result, cache_path)

        return result

    return wrapper
