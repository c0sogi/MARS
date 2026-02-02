import os
import sys
import random
import numpy as np
import torch
import logging
import pandas as pd
import time
from contextlib import contextmanager
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name="experiment", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger that writes to stdout and an optional file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
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


def calculate_accuracy(predictions, targets):
    """
    Calculates the exact string match accuracy between predictions and targets.

    Args:
        predictions (list or np.array): List of predicted strings.
        targets (list or np.array): List of ground truth strings.

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    # Convert to numpy arrays for efficient comparison if they aren't already
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    # Ensure string type
    predictions = predictions.astype(str)
    targets = targets.astype(str)

    correct = np.sum(predictions == targets)
    total = len(targets)

    return correct / total if total > 0 else 0.0


def ensure_dir(path):
    """
    Ensures that the directory for the given path exists.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a parquet file.
    Returns None if file does not exist.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_npy(arr, path):
    """
    Saves a numpy array to a .npy file.
    """
    ensure_dir(path)
    np.save(path, arr)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.
    Returns None if file does not exist.
    """
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)  # strictly no pickle
    return None


@contextmanager
def timer(name, logger=None):
    """
    Context manager to measure and log execution time.
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    msg = f"[{name}] done in {elapsed_time:.2f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)
