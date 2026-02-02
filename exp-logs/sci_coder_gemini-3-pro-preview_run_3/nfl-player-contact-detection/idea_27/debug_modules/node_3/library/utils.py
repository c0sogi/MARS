import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels (binary).

    Returns:
        float: MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check if inputs are empty
    if len(y_true) == 0:
        return 0.0

    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, steps=100):
    """
    Finds the optimal probability threshold that maximizes MCC.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_proba (array-like): Predicted probabilities.
        steps (int): Number of threshold steps to search.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    best_mcc = -1.0
    best_thresh = 0.5

    # Search range from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, steps)

    for thresh in thresholds:
        y_pred_bin = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_bin)

        if score > best_mcc:
            best_mcc = score
            best_thresh = thresh

    return best_thresh, best_mcc


def validate_schema(df, expected_columns, strict=True):
    """
    Validates that the DataFrame contains the expected columns.

    Args:
        df (pd.DataFrame): The dataframe to check.
        expected_columns (list): List of column names that must exist.
        strict (bool): If True, raises ValueError on missing columns.

    Returns:
        bool: True if valid, False otherwise (if strict is False).
    """
    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        msg = f"Schema Validation Failed. Missing columns: {missing_cols}"
        if strict:
            raise ValueError(msg)
        else:
            print(msg)
            return False

    # Check for nulls in expected columns (Warning only)
    if not df.empty:
        null_counts = df[expected_columns].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            # We print a warning but do not fail, as some models handle NaNs
            # or imputation might happen later.
            print(
                f"Schema Validation Warning: Null values found in columns: {cols_with_nulls.to_dict()}"
            )

    return True


def get_cache_path(base_name, params_dict=None, extension=".parquet"):
    """
    Generates a cache file path based on a base name and optional parameters hash.

    Args:
        base_name (str): Identifier for the file (e.g., 'features_streamA').
        params_dict (dict, optional): Dictionary of parameters to hash into the filename.
        extension (str): File extension (e.g., '.parquet', '.npy').

    Returns:
        str: Full path to the cache file.
    """
    if params_dict:
        # Create a deterministic string representation of the dictionary
        # sort_keys=True ensures consistent ordering
        param_str = json.dumps(params_dict, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()
        filename = f"{base_name}_{param_hash}{extension}"
    else:
        filename = f"{base_name}{extension}"

    return os.path.join(Config.CACHE_DIR, filename)


def save_cache(data, path):
    """
    Saves data to the cache path. Supports DataFrame (parquet) and numpy arrays (npy).

    Args:
        data: pd.DataFrame or np.ndarray
        path (str): Destination path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
    elif isinstance(data, np.ndarray):
        np.save(path, data)
    else:
        raise TypeError(
            f"Unsupported data type for caching: {type(data)}. Use pd.DataFrame or np.ndarray."
        )


def load_cache(path):
    """
    Loads data from the cache path.

    Args:
        path (str): Path to the cache file.

    Returns:
        pd.DataFrame or np.ndarray: Loaded data, or None if not found.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file extension for loading: {path}")


def check_cache_exists(path):
    """
    Checks if a cache file exists.

    Args:
        path (str): Path to check.

    Returns:
        bool: True if exists, False otherwise.
    """
    return os.path.exists(path)
