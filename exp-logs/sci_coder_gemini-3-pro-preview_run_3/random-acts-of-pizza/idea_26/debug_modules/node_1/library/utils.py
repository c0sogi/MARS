import os
import time
import random
import datetime
import warnings
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Also suppresses warnings to keep output clean.

    Args:
        seed (int): The seed value to use.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Python random
    random.seed(seed)

    # Environment variable for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_header(title):
    """
    Prints a formatted header block to distinguish sections of the log.

    Args:
        title (str): The title text to display.
    """
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def print_info(message):
    """
    Prints a message with a timestamp for logging purposes.

    Args:
        message (str): The message to print.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, description="Task"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print_info(f"Starting: {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print_info(
            f"Finished: {self.description}. Duration: {elapsed_time:.6f} seconds."
        )


def get_cached_data(cache_path, process_func, load_cached=True, **kwargs):
    """
    Generic caching mechanism for deterministic data processing.

    Logic:
    1. If load_cached is True and file exists, load and return.
    2. Otherwise, execute process_func(**kwargs).
    3. Save result to cache_path.
    4. Return result.

    Supported formats: .parquet (pandas DataFrame), .npy (numpy array), .npz (numpy dict).

    Args:
        cache_path (str): Full path to the cache file.
        process_func (callable): Function to compute data if cache is missing.
        load_cached (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to process_func.

    Returns:
        The loaded or computed data.
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if load_cached and os.path.exists(cache_path):
        print_info(f"Loading cached data from: {cache_path}")
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path)
            elif cache_path.endswith(".npz"):
                return np.load(cache_path)
            else:
                # Default to pickle if extension not recognized (though instructions prefer parquet/npy)
                # For safety given instructions, we raise error or assume user handles format inside function
                # But here we implement standard formats.
                raise ValueError(f"Unsupported cache file extension: {cache_path}")
        except Exception as e:
            print_info(f"Failed to load cache ({e}). Recomputing...")

    # Compute data
    print_info(f"Computing data for: {cache_path}")
    data = process_func(**kwargs)

    # Save data
    print_info(f"Saving data to cache: {cache_path}")
    if cache_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=False)
        else:
            raise TypeError("Data must be a pandas DataFrame for .parquet cache.")
    elif cache_path.endswith(".npy"):
        np.save(cache_path, data)
    elif cache_path.endswith(".npz"):
        if isinstance(data, dict):
            np.savez(cache_path, **data)
        else:
            # If not a dict, generic save
            np.savez(cache_path, data)
    else:
        raise ValueError(f"Unsupported cache file extension for saving: {cache_path}")

    return data
