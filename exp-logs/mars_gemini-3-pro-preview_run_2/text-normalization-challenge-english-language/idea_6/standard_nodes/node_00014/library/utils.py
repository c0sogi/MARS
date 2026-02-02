import os
import sys
import logging
import unicodedata
import torch
import pandas as pd
from library.config import Config, set_seed


def get_logger(name: str = Config.PROJECT_NAME) -> logging.Logger:
    """
    Configures and returns a logger that prints to stdout.
    Ensures handlers are not duplicated if get_logger is called multiple times.

    Args:
        name (str): The name of the logger. Defaults to the project name in Config.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

        # Prevent propagation to root logger to avoid double logging if root is configured
        logger.propagate = False

    return logger


def get_device() -> torch.device:
    """
    Retrieves the PyTorch device configured in the Config class.

    Returns:
        torch.device: The device (CPU or CUDA) to be used for computation.
    """
    return Config.DEVICE


def clean_text(text: str) -> str:
    """
    Performs standard text normalization and cleaning.
    Uses NFKC normalization to handle unicode characters (e.g., full-width digits)
    and strips leading/trailing whitespace.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        return str(text)

    # Normalize unicode characters (e.g., converting full-width '１' to '1')
    text = unicodedata.normalize("NFKC", text)

    # Strip whitespace
    text = text.strip()

    return text


def ensure_dir(path: str):
    """
    Ensures that the directory for the given file path exists.

    Args:
        path (str): The file path.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_cache(df: pd.DataFrame, path: str):
    """
    Saves a pandas DataFrame to a parquet file, ensuring the directory exists.
    Complies with the requirement to use parquet instead of pickle.

    Args:
        df (pd.DataFrame): The dataframe to save.
        path (str): The destination file path.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def load_cache(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file if it exists.

    Args:
        path (str): The file path to load from.

    Returns:
        pd.DataFrame or None: The loaded dataframe, or None if the file does not exist.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
