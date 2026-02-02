import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def setup_logger(
    name: str = "logger", log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, only logs to console.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error between true and predicted values.

    Args:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def save_submission(submission_df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the submission DataFrame to a CSV file.
    Ensures the output directory exists.

    Args:
        submission_df (pd.DataFrame): DataFrame containing 'segment_id' and 'time_to_eruption'.
        file_path (str): Full path where the CSV should be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    submission_df.to_csv(file_path, index=False)
    print(f"Submission saved to {file_path}")


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
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
                # Use float32 for compatibility with standard ML libraries
                # float16 can sometimes cause numerical instability
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(
            f"Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df
