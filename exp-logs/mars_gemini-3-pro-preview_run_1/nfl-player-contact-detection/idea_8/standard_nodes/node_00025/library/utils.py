import os
import sys
import gc
import json
import random
import hashlib
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # If torch is installed/used in the environment, seed it as well
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def setup_logger(
    name: str = "VRC_ME", log_file: str = "execution.log"
) -> logging.Logger:
    """
    Configures a logger to output to both console and a file in the working directory.

    Args:
        name: Name of the logger.
        log_file: Filename for the log output.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    log_path = os.path.join(Config.WORKING_DIR, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_experiment_hash(params: dict) -> str:
    """
    Generates a deterministic MD5 hash from a dictionary of parameters.
    Used for parameter-aware caching of feature datasets.

    Args:
        params: Dictionary of configuration parameters.

    Returns:
        str: MD5 hash string.
    """

    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    # Sort keys to ensure deterministic string representation
    param_str = json.dumps(params, sort_keys=True, default=default_serializer)
    return hashlib.md5(param_str.encode("utf-8")).hexdigest()


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        float: MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def garbage_collection():
    """
    Triggers explicit garbage collection to manage memory usage.
    """
    gc.collect()


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df: Input Pandas DataFrame.
        verbose: If True, prints memory reduction statistics.

    Returns:
        pd.DataFrame: Downcasted DataFrame.
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(
                        np.float32
                    )  # float16 has low precision, safe to use float32
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Memory usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def save_to_parquet(df: pd.DataFrame, filename: str):
    """
    Saves a DataFrame to Parquet format in the working directory.

    Args:
        df: DataFrame to save.
        filename: Name of the file (e.g., 'features.parquet').
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    path = os.path.join(Config.WORKING_DIR, filename)
    df.to_parquet(path, index=False)


def load_from_parquet(filename: str) -> pd.DataFrame:
    """
    Loads a DataFrame from Parquet format in the working directory.

    Args:
        filename: Name of the file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame if exists, else None.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
