import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
import functools
import hashlib
import json
from library import config


def setup_logging(log_file_path=None, level=logging.INFO):
    """
    Configures logging to output to both console and a file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed=config.SEED):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df, verbose=False):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

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
            f"Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def _hash_args(*args, **kwargs):
    """
    Helper to generate a consistent hash from function arguments.
    Filters out 'load_cached_data' and 'cache_path' to ensure the hash
    reflects the data generation parameters, not the I/O instructions.
    """
    # Filter out arguments that don't affect the computation result
    filter_keys = ["load_cached_data", "cache_path"]
    hash_dict = {
        "args": [str(a) for a in args],
        "kwargs": {k: str(v) for k, v in kwargs.items() if k not in filter_keys},
    }

    # Create a consistent string representation
    try:
        encoded = json.dumps(hash_dict, sort_keys=True).encode()
    except Exception:
        # Fallback for non-serializable objects
        encoded = str(hash_dict).encode()

    return hashlib.md5(encoded).hexdigest()


def cache_result(filename=None, file_format="parquet"):
    """
    Decorator to cache function results to disk (Parquet or NPY).

    Logic:
    1. Determine Cache Path:
       - Uses 'cache_path' from kwargs if provided.
       - Else uses 'filename' from decorator if provided.
       - Else generates a filename based on function name and argument hash.
    2. Load:
       - If 'load_cached_data' is True and file exists, load and return.
    3. Compute & Save:
       - Otherwise, execute function, save result, and return.

    Args:
        filename (str, optional): Default filename to use relative to config.CACHE_DIR.
        file_format (str): 'parquet' or 'npy'.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            load_cached_data = kwargs.get("load_cached_data", False)

            # 1. Determine Cache Path
            path = kwargs.get("cache_path")

            if path is None:
                if filename is not None:
                    path = os.path.join(config.CACHE_DIR, filename)
                else:
                    # Generate parameter-aware filename
                    arg_hash = _hash_args(*args, **kwargs)
                    ext = "npy" if file_format == "npy" else "parquet"
                    path = os.path.join(
                        config.CACHE_DIR, f"{func.__name__}_{arg_hash}.{ext}"
                    )

            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # 2. Try to Load
            if load_cached_data and os.path.exists(path):
                print(f"Loading cached data from {path}...")
                try:
                    if file_format == "parquet":
                        return pd.read_parquet(path)
                    elif file_format == "npy":
                        return np.load(path)
                    else:
                        raise ValueError(f"Unsupported format: {file_format}")
                except Exception as e:
                    print(f"Failed to load cache ({e}). Recomputing...")

            # 3. Compute
            print(f"Computing data for {func.__name__}...")
            result = func(*args, **kwargs)

            # 4. Save
            print(f"Saving cached data to {path}...")
            if file_format == "parquet":
                if isinstance(result, pd.DataFrame):
                    result.to_parquet(path, index=False)
                else:
                    raise TypeError(
                        "Result must be a pandas DataFrame for parquet format."
                    )
            elif file_format == "npy":
                if isinstance(result, np.ndarray):
                    np.save(path, result)
                else:
                    raise TypeError("Result must be a numpy ndarray for npy format.")
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            return result

        return wrapper

    return decorator
