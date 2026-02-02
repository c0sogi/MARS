import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
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


def setup_logger(name: str, log_file: str, level: int = logging.INFO):
    """
    Configures and returns a logger that writes to both a file and the console.

    Args:
        name (str): The name of the logger.
        log_file (str): The file path where logs should be saved.
        level (int): The logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_submission(request_ids, predictions, output_path: str):
    """
    Saves the submission file in the required format.

    Args:
        request_ids (array-like): List of request IDs.
        predictions (array-like): List of predicted probabilities.
        output_path (str): Path to save the CSV file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    df.to_csv(output_path, index=False)


def save_to_cache(data, path: str):
    """
    Saves data to a cache file. Supports .npy for numpy arrays and .parquet for pandas DataFrames.

    Args:
        data: The data to save (np.ndarray or pd.DataFrame).
        path (str): The destination file path.
    """
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(data, np.ndarray):
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)
    elif isinstance(data, pd.DataFrame):
        if not path.endswith(".parquet"):
            path += ".parquet"
        data.to_parquet(path, index=False)
    else:
        raise ValueError(
            "Unsupported data type for caching. Use numpy array or pandas DataFrame."
        )


def load_from_cache(path: str):
    """
    Loads data from a cache file. Supports .npy and .parquet.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded data (np.ndarray or pd.DataFrame), or None if file does not exist.
    """
    # Handle implicit extensions if necessary, though strict paths are preferred
    if not os.path.exists(path):
        # Try appending extensions if not present
        if os.path.exists(path + ".npy"):
            path += ".npy"
        elif os.path.exists(path + ".parquet"):
            path += ".parquet"
        else:
            return None

    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension for loading: {path}")
