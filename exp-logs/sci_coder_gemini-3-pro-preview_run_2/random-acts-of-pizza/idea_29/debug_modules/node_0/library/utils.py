import os
import random
import logging
import numpy as np
import torch
import joblib

# Define the working directory for this specific idea iteration
WORKING_DIR = "./working/idea_29/"


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_object(obj, file_path):
    """
    Saves a Python object to a file using joblib.
    Ensures the directory exists before saving.

    Args:
        obj (Any): The object to save.
        file_path (str): The destination file path.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (str): The source file path.

    Returns:
        Any: The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return joblib.load(file_path)


def get_device():
    """
    Returns the appropriate torch device (cuda if available, else cpu).

    Returns:
        torch.device: The device to use for tensor computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
