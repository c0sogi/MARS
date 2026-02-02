import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def setup_logger(name: str = "TextNorm", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_submission_id(sentence_id, token_id):
    """
    Formats the sentence and token IDs into the submission ID string.

    Args:
        sentence_id: The ID of the sentence.
        token_id: The ID of the token within the sentence.

    Returns:
        str: Formatted ID (e.g., "123_5").
    """
    return f"{sentence_id}_{token_id}"


def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """
    Calculates the exact match accuracy between true and predicted tokens.

    Args:
        y_true (list): List of ground truth strings.
        y_pred (list): List of predicted strings.

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    if len(y_true) == 0:
        return 0.0

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def load_or_process(path: str, process_fn, load_cached_data: bool = True, **kwargs):
    """
    Implements the deterministic caching logic.

    1. Checks if `load_cached_data` is True and the file exists.
    2. If yes, loads and returns the data.
    3. If no, calls `process_fn(**kwargs)`, saves the result to `path`, and returns it.

    Supported formats: .parquet (pandas DataFrame), .npy (numpy array/dict).

    Args:
        path (str): The full path to the cache file.
        process_fn (callable): The function to generate data if cache is missed.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to `process_fn`.

    Returns:
        The loaded or processed data.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    file_ext = os.path.splitext(path)[1].lower()

    # Try to load
    if load_cached_data and os.path.exists(path):
        try:
            if file_ext == ".parquet":
                print(f"Loading cached data from {path}...")
                return pd.read_parquet(path)
            elif file_ext == ".npy":
                print(f"Loading cached data from {path}...")
                return np.load(path, allow_pickle=True).item()
            else:
                print(f"Warning: Unsupported cache format {file_ext}. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache from {path}: {e}. Recomputing...")

    # Compute
    print(f"Computing data (Cache miss or force refresh)...")
    data = process_fn(**kwargs)

    # Save
    print(f"Saving data to {path}...")
    if file_ext == ".parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise TypeError("Data must be a pandas DataFrame for .parquet cache.")
    elif file_ext == ".npy":
        np.save(path, data)
    else:
        raise ValueError(
            f"Unsupported file extension for caching: {file_ext}. Use .parquet or .npy"
        )

    return data
