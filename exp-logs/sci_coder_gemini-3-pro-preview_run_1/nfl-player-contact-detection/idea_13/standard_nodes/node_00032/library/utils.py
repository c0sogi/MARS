import os
import random
import numpy as np
import torch
import pandas as pd
import hashlib
import json
from sklearn.metrics import matthews_corrcoef
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): Whether to print the memory reduction statistics.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generates a unique MD5 hash based on the provided arguments.
    Useful for creating cache filenames that change when parameters change.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        str: MD5 hash string.
    """

    # Helper to handle non-serializable objects by converting to string
    def default_serializer(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    # Create a list of all inputs
    key_data = {"args": args, "kwargs": kwargs}

    # Serialize to JSON string with sorting to ensure determinism
    try:
        serialized = json.dumps(key_data, sort_keys=True, default=default_serializer)
    except TypeError:
        # Fallback if json fails
        serialized = str(key_data)

    # Generate MD5 hash
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
