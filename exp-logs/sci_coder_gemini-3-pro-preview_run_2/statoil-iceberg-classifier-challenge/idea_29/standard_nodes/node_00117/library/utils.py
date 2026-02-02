import os
import sys
import logging
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Delegates to the implementation in config.py to avoid code duplication.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    config.set_seed(seed)


def get_logger(name, log_filename="run.log"):
    """
    Configures and returns a logger that writes to console and a file.

    Args:
        name (str): Name of the logger.
        log_filename (str): Name of the log file. Defaults to "run.log".
                            The file is saved in config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplication if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_path = os.path.join(config.WORKING_DIR, log_filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
