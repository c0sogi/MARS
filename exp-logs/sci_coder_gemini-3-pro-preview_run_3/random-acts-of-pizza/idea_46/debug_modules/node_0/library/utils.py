import os
import sys
import time
import random
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score

# Import configuration
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Handle PyTorch seeding if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores or probabilities.

    Returns:
        float: The computed AUC score.
    """
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in y_true during debugging/small batches
        score = 0.5
    return score


def log_info(message: str):
    """
    Prints a message with a timestamp.

    Args:
        message (str): The message to log.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


@contextmanager
def timer(task_name: str):
    """
    Context manager to measure and log the execution time of a block of code.

    Args:
        task_name (str): Description of the task being timed.
    """
    log_info(f"Starting {task_name}...")
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        log_info(f"Finished {task_name}. Duration: {t1 - t0:.4f} seconds")


def load_data(split: str):
    """
    Loads the specified dataset split from the metadata parquet files.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If an invalid split name is provided.
    """
    if split == "train":
        path = Config.TRAIN_PATH
    elif split == "val":
        path = Config.VAL_PATH
    elif split == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(
            f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_parquet(path)

    # Debugging: Sample data if configured
    if Config.DEBUG and split == "train":
        log_info(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from {split}")
        if len(df) > Config.DEBUG_SAMPLE_SIZE:
            df = df.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.RANDOM_SEED
            ).reset_index(drop=True)

    return df


def save_cache(data, filename: str, use_parquet: bool = False):
    """
    Saves data to the cache directory defined in Config.
    Supports .npy for numpy arrays and .parquet for pandas DataFrames.

    Args:
        data: The object to save (numpy array or pandas DataFrame).
        filename (str): The name of the file (e.g., 'features.npy').
        use_parquet (bool): If True, saves as parquet (requires DataFrame).
                            If False, saves as numpy (requires Array).
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, filename)

    log_info(f"Caching data to {file_path}...")

    if use_parquet:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame to save as parquet.")
        data.to_parquet(file_path, index=False)
    else:
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                "Data is a DataFrame but use_parquet=False. Use use_parquet=True or convert to numpy."
            )
        np.save(file_path, data)


def load_cache(filename: str, use_parquet: bool = False):
    """
    Loads data from the cache directory if it exists.

    Args:
        filename (str): The name of the file to load.
        use_parquet (bool): If True, loads as parquet. If False, loads as numpy.

    Returns:
        The loaded data, or None if the file does not exist.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)

    if not os.path.exists(file_path):
        return None

    log_info(f"Loading cached data from {file_path}...")

    if use_parquet:
        return pd.read_parquet(file_path)
    else:
        return np.load(file_path, allow_pickle=False)
