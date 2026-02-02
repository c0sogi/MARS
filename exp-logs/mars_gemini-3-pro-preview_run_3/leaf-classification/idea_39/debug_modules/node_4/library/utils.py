import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import SEED, CACHE_DIR


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    log_file: str = "execution.log", level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        log_file (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a custom logger
    logger = logging.getLogger("project_logger")
    logger.setLevel(level)

    # Avoid adding handlers multiple times if the logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    # Using a simple format: Timestamp - Level - Message
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_array(data: np.ndarray, filename: str, sub_dir: str = None):
    """
    Saves a numpy array to the cache directory.

    Args:
        data (np.ndarray): The array to save.
        filename (str): The filename (e.g., 'features.npy').
        sub_dir (str, optional): Subdirectory within CACHE_DIR.
    """
    if sub_dir:
        target_dir = os.path.join(CACHE_DIR, sub_dir)
    else:
        target_dir = CACHE_DIR

    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)
    np.save(file_path, data)


def load_array(filename: str, sub_dir: str = None) -> np.ndarray:
    """
    Loads a numpy array from the cache directory.

    Args:
        filename (str): The filename to load.
        sub_dir (str, optional): Subdirectory within CACHE_DIR.

    Returns:
        np.ndarray: The loaded array, or None if file does not exist.
    """
    if sub_dir:
        target_dir = os.path.join(CACHE_DIR, sub_dir)
    else:
        target_dir = CACHE_DIR

    file_path = os.path.join(target_dir, filename)

    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=False)
    else:
        return None
