import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_common_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: list = None
) -> list:
    """
    Identifies the intersection of columns between train and test DataFrames.
    Useful for preventing feature leakage by ensuring only features present in both sets are used.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        exclude_cols: List of columns to exclude from the intersection (e.g., targets).

    Returns:
        List of common column names sorted alphabetically.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    common_cols = list(train_cols.intersection(test_cols))

    # Filter out excluded columns
    final_cols = [col for col in common_cols if col not in exclude_cols]

    # Sort for deterministic behavior
    final_cols.sort()

    return final_cols


def save_parquet(df: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to a parquet file, creating the directory if it doesn't exist.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def save_numpy(data, path: str, compressed: bool = True):
    """
    Saves a numpy array or dictionary of arrays to a .npy or .npz file.

    Args:
        data: Numpy array or dictionary of numpy arrays.
        path: Destination path.
        compressed: Whether to use compression for .npz files.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if path.endswith(".npz"):
        if isinstance(data, dict):
            if compressed:
                np.savez_compressed(path, **data)
            else:
                np.savez(path, **data)
        else:
            if compressed:
                np.savez_compressed(path, data=data)
            else:
                np.savez(path, data=data)
    else:
        # Default to .npy if extension is missing or explicitly .npy
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)


def load_numpy(path: str):
    """
    Loads a numpy array or dictionary of arrays from a .npy or .npz file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")

    # allow_pickle=True is required for loading object arrays or .npz files correctly,
    # even if we avoid using pickle for serialization logic.
    return np.load(path, allow_pickle=True)
