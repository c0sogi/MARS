import os
import sys
import random
import time
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CuDNN backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def print_header(title):
    """
    Prints a formatted header to separate sections of the log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*80}")
    print(f"[{timestamp}] {title}")
    print(f"{'='*80}")


def print_info(message):
    """
    Prints an informational message with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")


def print_metric(name, value):
    """
    Prints a metric name and its value with full precision.
    """
    # Using str(value) ensures full precision is displayed without rounding
    print(f"[METRIC] {name}: {value}")


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, description):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print_info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        # Format time as H:M:S or seconds depending on duration
        if elapsed_time < 60:
            duration_str = f"{elapsed_time:.4f} seconds"
        else:
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            duration_str = f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

        print_info(f"Finished: {self.description} | Duration: {duration_str}")


def get_device():
    """
    Checks availability of GPU and returns the torch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print_info(f"GPU Available: {device_name}")
    else:
        device = torch.device("cpu")
        print_info("GPU Not Available. Using CPU.")
    return device


def save_to_parquet(df, path):
    """
    Helper to save DataFrame to parquet, ensuring directory exists.
    Adheres to the requirement of not using pickle.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path, index=False)
    print_info(f"Saved parquet to {path}")


def load_from_parquet(path):
    """
    Helper to load DataFrame from parquet.
    Returns None if file does not exist.
    """
    if os.path.exists(path):
        print_info(f"Loading parquet from {path}")
        return pd.read_parquet(path)
    return None


def save_numpy(array, path):
    """
    Helper to save numpy array, ensuring directory exists.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, array)
    print_info(f"Saved numpy array to {path}")


def load_numpy(path):
    """
    Helper to load numpy array.
    Returns None if file does not exist.
    """
    if os.path.exists(path):
        print_info(f"Loading numpy array from {path}")
        return np.load(path)
    return None
