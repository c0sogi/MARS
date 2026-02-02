import os
import sys
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def setup_logging(log_file=None, level=logging.INFO):
    """
    Configures the logging module to output to stdout and optionally a file.

    Args:
        log_file (str, optional): Path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logger: Configured logger instance.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure directory exists for log file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Force reconfiguration if already configured
    )
    return logging.getLogger()


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def memory_reduction(df):
    """
    Iterates through a DataFrame's columns and modifies their data types
    to reduce memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Downcast float64 to float32. Float16 is often too imprecise for physics calculations.
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    # Logging is not initialized inside this function to keep it pure,
    # but in a real scenario, we might log the reduction.
    return df


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name="Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Done in {elapsed_time:.2f} s")
