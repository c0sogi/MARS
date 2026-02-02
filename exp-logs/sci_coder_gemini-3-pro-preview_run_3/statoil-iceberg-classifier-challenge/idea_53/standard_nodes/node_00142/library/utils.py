import os
import sys
import logging
from library.config import set_seed as _set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation provided in library.config.

    Args:
        seed (int): The seed value to use.
    """
    _set_seed(seed)


def get_logger(name, log_file=None):
    """
    Configures and returns a logger instance that writes to stdout and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): The path to the log file. If provided, the directory
                                  will be created if it does not exist.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Prevent propagation to root logger to avoid double logging if root is configured
    logger.propagate = False

    return logger
