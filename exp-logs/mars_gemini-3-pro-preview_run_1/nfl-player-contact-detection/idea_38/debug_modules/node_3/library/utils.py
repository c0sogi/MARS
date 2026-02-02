import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def setup_logger(name="exp_logger", log_file="execution.log"):
    """
    Sets up a logger that writes to both console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    log_path = os.path.join(Config.CACHE_DIR, log_file)
    # Ensure directory exists for the log file
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def reduce_mem_usage(df, verbose=True):
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
                # Use float32 for safety, float16 can be unstable in some libs
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


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.
    """
    return matthews_corrcoef(y_true, y_pred)


def _get_path(filename):
    """
    Helper to resolve path relative to CACHE_DIR if not absolute.
    """
    if os.path.isabs(filename):
        return filename
    return os.path.join(Config.CACHE_DIR, filename)


def save_joblib(obj, filename):
    """
    Saves a python object using joblib.
    """
    filepath = _get_path(filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    # print(f"Saved object to {filepath}")


def load_joblib(filename):
    """
    Loads a python object using joblib. Returns None if file not found.
    """
    filepath = _get_path(filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


def save_parquet(df, filename):
    """
    Saves a pandas DataFrame to parquet.
    """
    filepath = _get_path(filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    # print(f"Saved dataframe to {filepath}")


def load_parquet(filename):
    """
    Loads a pandas DataFrame from parquet. Returns None if file not found.
    """
    filepath = _get_path(filename)
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def save_npy(arr, filename):
    """
    Saves a numpy array to .npy file.
    """
    filepath = _get_path(filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, arr)
    # print(f"Saved numpy array to {filepath}")


def load_npy(filename):
    """
    Loads a numpy array from .npy file. Returns None if file not found.
    """
    filepath = _get_path(filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    return None
