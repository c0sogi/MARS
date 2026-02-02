import os
import logging
import hashlib
import joblib
import numpy as np
import pandas as pd
from library.config import PathConfig


def setup_logging(log_path=None, level=logging.INFO):
    """
    Configures the logging module to output to console and optionally a file.

    Args:
        log_path (str, optional): Path to the log file.
        level (int): Logging level (default: logging.INFO).
    """
    handlers = [logging.StreamHandler()]

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def save_artifact(obj, path):
    """
    Saves a general Python object (e.g., model, dict) using joblib.

    Args:
        obj: The object to save.
        path (str): Destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logging.info(f"Artifact saved to {path}")


def load_artifact(path):
    """
    Loads a general Python object using joblib.

    Args:
        path (str): Path to the artifact.

    Returns:
        The loaded object, or None if file does not exist.
    """
    if not os.path.exists(path):
        logging.warning(f"Artifact not found at {path}")
        return None
    return joblib.load(path)


def save_dataframe(df, path):
    """
    Saves a pandas DataFrame to Parquet format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): Destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info(f"DataFrame saved to {path}")


def load_dataframe(path):
    """
    Loads a pandas DataFrame from Parquet format.

    Args:
        path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if not found.
    """
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def save_numpy(arr, path):
    """
    Saves a numpy array to .npy format.

    Args:
        arr (np.ndarray): The array to save.
        path (str): Destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    logging.info(f"Numpy array saved to {path}")


def load_numpy(path):
    """
    Loads a numpy array from .npy format.

    Args:
        path (str): Path to the .npy file.

    Returns:
        np.ndarray: The loaded array, or None if not found.
    """
    if not os.path.exists(path):
        return None
    return np.load(path)


def generate_cache_key(*args, **kwargs):
    """
    Generates a deterministic MD5 hash based on input arguments.
    Useful for creating unique filenames for cached data based on parameters.

    Args:
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.

    Returns:
        str: MD5 hash string.
    """
    # Convert args and kwargs to a sorted string representation
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def calculate_distance(x1, y1, x2, y2):
    """
    Vectorized calculation of Euclidean distance between two sets of coordinates.

    Args:
        x1, y1: Coordinates of first entity (scalar or array).
        x2, y2: Coordinates of second entity (scalar or array).

    Returns:
        np.ndarray or float: Euclidean distance.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_vector_magnitude(vx, vy):
    """
    Vectorized calculation of vector magnitude (speed/acceleration).

    Args:
        vx: X-component of vector.
        vy: Y-component of vector.

    Returns:
        np.ndarray or float: Magnitude.
    """
    return np.sqrt(vx**2 + vy**2)


def reduce_mem_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with optimized types.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
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
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(
                        np.float32
                    )  # float16 has low precision, using float32 is safer
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")

    return df
