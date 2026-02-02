import os
import sys
import logging
import torch
import pandas as pd
import numpy as np

# Import from library.config to ensure consistency and avoid re-implementation
from library.config import seed_everything as _seed_everything
from library.config import DEVICE


def seed_everything(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in library.config.
    """
    _seed_everything(seed)


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) as determined in library.config.
    """
    return DEVICE


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training loops.
    """

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.val} (Avg: {self.avg})"


def get_logger(log_file_path):
    """
    Sets up a logger that writes to both a file and the console.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logs
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def print_full_precision_metrics(metrics, phase="Validation"):
    """
    Prints metrics with full floating-point precision.

    Args:
        metrics (dict): Dictionary of metric names and their values.
        phase (str): The phase of execution (e.g., 'Train', 'Validation').
    """
    print(f"\n[{phase} Metrics]")
    for k, v in metrics.items():
        # Printing without formatting strings preserves full precision
        print(f"{k}: {v}")


def save_checkpoint(state, is_best, save_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        save_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_filepath)


def load_or_process_cache(cache_path, process_fn, load_cached_data=True, **kwargs):
    """
    Generic utility for caching deterministic data processing results.

    Args:
        cache_path (str): Path to the cache file (must be .parquet or .npy).
        process_fn (callable): Function to execute if cache is missing or reload is forced.
        load_cached_data (bool): If True, attempts to load from cache first.
        **kwargs: Keyword arguments passed to process_fn.

    Returns:
        The loaded or computed data.
    """
    # Ensure the directory for the cache file exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=True)
            else:
                raise ValueError("Unsupported cache format. Use .parquet or .npy")
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing...")

    # 2. Compute data if cache missed or failed
    print("Computing data...")
    data = process_fn(**kwargs)

    # 3. Save to cache
    print(f"Saving data to cache at {cache_path}...")
    if cache_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path)
        else:
            raise TypeError("Data must be a pandas DataFrame to save as parquet.")
    elif cache_path.endswith(".npy"):
        np.save(cache_path, data)
    else:
        raise ValueError("Unsupported cache format. Use .parquet or .npy")

    return data
