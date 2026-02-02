import os
import random
import numpy as np
import torch
import hashlib
import json
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, steps=100):
    """
    Finds the optimal probability threshold that maximizes MCC.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_proba (np.array): Predicted probabilities.
        steps (int): Number of steps to search between 0 and 1.

    Returns:
        tuple: (best_threshold, best_score)
    """
    best_score = -1.0
    best_thresh = 0.5

    # Search range from 0.01 to 0.99 to avoid edge cases
    thresholds = np.linspace(0.01, 0.99, steps)

    for thresh in thresholds:
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def generate_config_hash(config_dict):
    """
    Generates a deterministic SHA-256 hash from a configuration dictionary.
    Useful for cache invalidation.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: Hexadecimal hash string.
    """
    # Sort keys to ensure deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def get_cache_path(base_dir, prefix, config_dict, extension):
    """
    Constructs a cache file path based on configuration hash.

    Args:
        base_dir (str): Directory to store the cache.
        prefix (str): Identifier for the data (e.g., 'train_features').
        config_dict (dict): Configuration used to generate the data.
        extension (str): File extension (e.g., 'parquet', 'npy').

    Returns:
        str: Full path to the cache file.
    """
    cfg_hash = generate_config_hash(config_dict)
    filename = f"{prefix}_{cfg_hash}.{extension.lstrip('.')}"
    return os.path.join(base_dir, filename)


def save_to_cache(data, path):
    """
    Saves data to cache using format inferred from extension.
    Supports .parquet (DataFrame) and .npy (NumPy array).

    Args:
        data: The data object to save.
        path (str): Destination file path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame for .parquet extension.")
    elif path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            raise ValueError("Data must be a numpy ndarray for .npy extension.")
    else:
        raise ValueError(f"Unsupported file extension for caching: {path}")


def load_from_cache(path):
    """
    Loads data from cache using format inferred from extension.

    Args:
        path (str): Path to the cache file.

    Returns:
        The loaded data (DataFrame or ndarray), or None if file does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for loading: {path}")
