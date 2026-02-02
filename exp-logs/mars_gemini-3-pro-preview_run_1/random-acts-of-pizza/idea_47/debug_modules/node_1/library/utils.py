import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from library.config import RANDOM_STATE


def seed_everything(seed: int = RANDOM_STATE):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to RANDOM_STATE from config.
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


def save_pickle(obj, path: str):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj: The Python object to save.
        path (str): The destination file path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_intersection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = None,
    exclude_cols: list = None,
):
    """
    Identifies the intersection of columns between train and test dataframes,
    excluding the target column and any other specified columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        target_col (str, optional): Name of the target column to exclude.
        exclude_cols (list, optional): List of additional columns to exclude (e.g., IDs).

    Returns:
        list: Sorted list of common feature names present in both dataframes.
    """
    # Get sets of columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Find intersection
    common_cols = train_cols.intersection(test_cols)

    # Prepare exclusions
    exclusions = set()
    if target_col:
        exclusions.add(target_col)
    if exclude_cols:
        exclusions.update(exclude_cols)

    # Remove exclusions and sort
    final_cols = list(common_cols - exclusions)
    final_cols.sort()

    return final_cols
