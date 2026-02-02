import os
import sys
import logging
from library.config import set_seed, Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Wraps the implementation provided in library.config to avoid code duplication.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    set_seed(seed)


def get_logger(name="TS-WBN", log_file=None):
    """
    Creates and configures a logger with both console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If provided, logs will also be written to this file.
                                  The directory for the log file will be created if it doesn't exist.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
