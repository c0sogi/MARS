import os
import sys
import random
import logging
import zlib
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger


def safe_hash(text, num_buckets=10000):
    """
    Deterministic hashing of a string to an integer within a range.
    Uses zlib.crc32 for consistency across runs/platforms.

    Args:
        text (str): Input text to hash.
        num_buckets (int): The modulus for the hash (vocabulary size for hashing).

    Returns:
        int: The hashed integer value.
    """
    if not isinstance(text, str):
        text = str(text)

    # crc32 is deterministic and fast
    # & 0xffffffff is to ensure unsigned 32-bit integer compatibility across Python versions
    hash_val = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
    return hash_val % num_buckets


def clean_text(text):
    """
    Ensures the input is a valid string.

    Args:
        text: Input object (str, int, float, etc.).

    Returns:
        str: String representation.
    """
    if text is None:
        return ""
    return str(text)


def load_dataset(path, dtype=None):
    """
    Loads a dataset with specific settings for the Text Normalization task.
    Ensures 'null', 'nan' etc. are read as strings, not NaNs.

    Args:
        path (str): Path to the CSV file.
        dtype (dict, optional): Dictionary of column types.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Default dtypes if not provided
    if dtype is None:
        dtype = {
            "sentence_id": "int32",
            "token_id": "int32",
            "class": "category",
            "before": "object",
            "after": "object",
            "id": "object",
        }

    # Only use dtypes that exist in the columns (handled by pandas, but good to be aware)
    # keep_default_na=False is crucial for this dataset
    df = pd.read_csv(
        path, dtype=dtype, keep_default_na=False, quoting=0  # csv.QUOTE_MINIMAL
    )

    return df
