import os
import random
import numpy as np
import torch
from library.config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # OS Environment for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes to prevent data leakage.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The testing dataframe.
        exclude_cols (list, optional): A list of column names to explicitly exclude
                                       from the intersection (e.g., target variables).

    Returns:
        list: A sorted list of column names present in both dataframes.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Find intersection
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    # Filter out excluded columns
    final_cols = [col for col in common_cols if col not in exclude_cols]

    # Return sorted list for consistent ordering
    return sorted(final_cols)
