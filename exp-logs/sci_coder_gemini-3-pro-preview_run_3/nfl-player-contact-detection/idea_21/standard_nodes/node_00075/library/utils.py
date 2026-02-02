import os
import random
import numpy as np
import pandas as pd
import hashlib
import json
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def reduce_mem_usage(df):
    """
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory usage. Prioritizes float32 over float16 to maintain
    precision for physics calculations.

    Args:
        df (pd.DataFrame): The dataframe to optimize.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Use float32 as minimum precision to avoid instability in physics features (jerk/acceleration)
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if col_type == object:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype("category")

    return df


def get_hashed_cache_path(base_name, config_dict, extension=".parquet"):
    """
    Generates a unique file path based on the hash of the configuration dictionary.
    Used to implement cache invalidation for feature engineering steps.

    Args:
        base_name (str): The prefix for the filename (e.g., 'features_streamA').
        config_dict (dict): Dictionary containing configuration parameters affecting the file.
        extension (str): File extension (default: .parquet).

    Returns:
        str: Full path to the cached file in the configured working directory.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Serialize config to JSON string with sorted keys for consistency
    # default=str handles non-serializable types safely
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Generate MD5 hash
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    # Construct filename
    filename = f"{base_name}_{config_hash}{extension}"

    return os.path.join(Config.WORKING_DIR, filename)
