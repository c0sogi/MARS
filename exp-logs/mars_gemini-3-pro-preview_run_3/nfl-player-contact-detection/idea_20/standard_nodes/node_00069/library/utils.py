import pandas as pd
import numpy as np
import hashlib
import logging
import sys
import os
from sklearn.metrics import matthews_corrcoef


def get_dataframe_hash(df):
    """
    Generates a deterministic SHA256 hash for a pandas DataFrame.
    Used for cache invalidation strategies.

    Args:
        df (pd.DataFrame): The dataframe to hash.

    Returns:
        str: Hex digest of the hash.
    """
    # Efficiently hash the data values using pandas utility
    # index=True ensures the index is part of the hash
    row_hashes = pd.util.hash_pandas_object(df, index=True).values

    # Create a string representation of columns and dtypes to catch schema changes
    schema_str = str(df.columns.tolist()) + str(df.dtypes.tolist()) + str(df.shape)

    # Combine content hash and schema hash
    hasher = hashlib.sha256()
    hasher.update(row_hashes.tobytes())
    hasher.update(schema_str.encode("utf-8"))

    return hasher.hexdigest()


def setup_logging(log_file_path):
    """
    Configures the root logger to write to both a file and stdout.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        logging.Logger: The configured root logger.
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def calc_mcc(y_true, y_pred_proba, threshold=0.5):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: The MCC score.
    """
    # Convert probabilities to binary class predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate MCC
    return matthews_corrcoef(y_true, y_pred)


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): Input dataframe.
        verbose (bool): Whether to print reduction statistics.

    Returns:
        pd.DataFrame: Dataframe with optimized types.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip object and category columns
        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            # Integer types
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            # Float types
            else:
                # Use float32 as standard for ML to avoid float16 precision issues
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
