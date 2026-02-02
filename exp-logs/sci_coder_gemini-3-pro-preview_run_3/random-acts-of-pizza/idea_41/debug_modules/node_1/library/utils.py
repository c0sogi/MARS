import os
import sys
import random
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from library.config import Config

# Handle optional torch import
try:
    import torch
except ImportError:
    torch = None


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name="Pipeline"):
    """
    Returns a simple logging function that prints to stdout with timestamps.
    """

    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{name}] {message}")
        sys.stdout.flush()

    return log


def load_metadata_splits():
    """
    Loads the train, validation, and test splits from the metadata directory.
    """
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    return train_df, val_df, test_df


def save_model(model, filename):
    """
    Saves a model or transformer using joblib to the working directory.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filename):
    """
    Loads a model or transformer using joblib from the working directory.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def save_cache_parquet(df, filename):
    """
    Saves a DataFrame to parquet in the working directory for caching.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_cache_parquet(filename):
    """
    Loads a DataFrame from parquet in the working directory. Returns None if not found.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def save_cache_npy(arr, filename):
    """
    Saves a numpy array to .npy in the working directory for caching.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, arr)


def load_cache_npy(filename):
    """
    Loads a numpy array from .npy in the working directory. Returns None if not found.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    return None


def save_cache_npz(data_dict, filename):
    """
    Saves a dictionary of arrays to .npz in the working directory for caching.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, **data_dict)


def load_cache_npz(filename):
    """
    Loads a .npz file from the working directory. Returns None if not found.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    return None


def print_metrics(metrics_dict, prefix="Val"):
    """
    Prints validation metrics with full precision.
    """
    # Using str(v) ensures full precision is printed without formatting
    metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics_dict.items()])
    print(f"[{prefix}] {metrics_str}")


class Timer:
    """
    Context manager for timing execution blocks.
    """

    def __init__(self, name="Task"):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        duration = self.end - self.start
        print(f"[{self.name}] Execution time: {duration:.6f} seconds")
