import os
import sys
import random
import numpy as np
import pandas as pd
import hashlib
import json
import functools
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def setup_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Try setting torch seed if available, but don't crash if not
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba):
    """
    Finds the optimal probability threshold that maximizes MCC.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities.

    Returns:
        best_threshold: The threshold value giving the highest MCC.
        best_score: The highest MCC score achieved.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1.0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def get_config_hash():
    """
    Generates an MD5 hash of the current Config attributes to detect changes.
    Filters out private attributes and methods.
    """
    config_dict = {}
    for key in dir(Config):
        if key.startswith("__"):
            continue
        value = getattr(Config, key)
        if callable(value):
            continue
        # Convert non-serializable types to string representation
        if isinstance(value, (set, tuple)):
            value = list(value)
        config_dict[key] = value

    # Sort keys to ensure deterministic JSON serialization
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def _save_data(data, path):
    """Helper to save data based on file extension."""
    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError(f"Expected DataFrame for parquet path: {path}")
    elif path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            raise ValueError(f"Expected ndarray for npy path: {path}")
    else:
        raise ValueError(f"Unsupported file extension for path: {path}")


def _load_data(path):
    """Helper to load data based on file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for path: {path}")


def use_config_cache(cache_keys):
    """
    Decorator to cache function results based on Config state.

    Args:
        cache_keys (list of str): Keys in Config.CACHE_FILES where results should be stored.
                                  The function must return a tuple of results matching the order of keys.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            load_cached_data = kwargs.get("load_cached_data", True)

            # Identify file paths from Config
            file_paths = [Config.CACHE_FILES[k] for k in cache_keys]

            # Determine hash file path (based on the first file's directory and name)
            base_dir = os.path.dirname(file_paths[0])
            base_name = os.path.basename(file_paths[0]).split(".")[0]
            hash_path = os.path.join(base_dir, f"{base_name}.hash")

            current_hash = get_config_hash()
            cache_valid = False

            # Check if cache exists and is valid
            if load_cached_data and os.path.exists(hash_path):
                # Verify all data files exist
                if all(os.path.exists(fp) for fp in file_paths):
                    try:
                        with open(hash_path, "r") as f:
                            stored_hash = f.read().strip()
                        if stored_hash == current_hash:
                            cache_valid = True
                    except Exception:
                        pass  # Hash file unreadable

            if cache_valid:
                # print(f"Loading cached data for {func.__name__}...")
                results = []
                for fp in file_paths:
                    results.append(_load_data(fp))

                # If single result, return it directly, else return tuple
                if len(results) == 1:
                    return results[0]
                return tuple(results)

            else:
                # print(f"Cache invalid or missing for {func.__name__}. Computing...")
                # Compute
                results = func(*args, **kwargs)

                # Ensure directory exists
                os.makedirs(base_dir, exist_ok=True)

                # Normalize results to a list
                if len(file_paths) == 1:
                    data_to_save = [results]
                else:
                    data_to_save = results
                    if len(data_to_save) != len(file_paths):
                        raise ValueError(
                            f"Function {func.__name__} returned {len(data_to_save)} items, but {len(file_paths)} cache keys were provided."
                        )

                # Save data
                for data, fp in zip(data_to_save, file_paths):
                    _save_data(data, fp)

                # Save hash
                with open(hash_path, "w") as f:
                    f.write(current_hash)

                return results

        return wrapper

    return decorator
