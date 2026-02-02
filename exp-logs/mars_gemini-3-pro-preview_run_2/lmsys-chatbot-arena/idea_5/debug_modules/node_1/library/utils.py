import os
import sys
import logging
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates the actual implementation to the Config class to avoid code duplication.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    Config.seed_everything(seed)


def get_logger(name: str = "main", log_file: str = None):
    """
    Creates and configures a logger that outputs to the console and optionally to a file.

    Args:
        name (str): The name of the logger instance. Defaults to "main".
        log_file (str, optional): The file path where logs should be saved.
                                  If None, logs are only printed to the console.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 1. Console Handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
