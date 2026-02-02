import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import RANDOM_STATE, DEVICE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) based on availability.

    Returns:
        torch.device: The device object to be used for tensor operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test DataFrames
    to prevent feature leakage.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        exclude_cols (list, optional): List of columns to exclude from the intersection
                                       (e.g., target variables, IDs).

    Returns:
        list: Sorted list of common column names.
    """
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    if exclude_cols:
        common_cols = common_cols - set(exclude_cols)

    return sorted(list(common_cols))
