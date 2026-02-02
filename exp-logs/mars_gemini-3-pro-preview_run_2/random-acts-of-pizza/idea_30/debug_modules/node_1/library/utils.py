import os
import sys
import logging
import joblib
from library.config import set_seed


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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


def save_object(obj, filepath):
    """
    Saves a Python object to a file using joblib.

    Args:
        obj: The Python object to save.
        filepath (str): The path where the object should be saved.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, filepath)
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise


def load_object(filepath):
    """
    Loads a Python object from a file using joblib.

    Args:
        filepath (str): The path to the file to load.

    Returns:
        The loaded Python object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise
