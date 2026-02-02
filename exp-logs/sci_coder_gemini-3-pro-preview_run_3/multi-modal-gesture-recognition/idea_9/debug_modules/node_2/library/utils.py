import logging
import os
import sys
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. If None, uses the default
                              seed defined in Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    # Delegate to the implementation in Config to avoid re-implementing logic
    Config.set_seed(seed)


def setup_logger(log_file=None, level=logging.INFO, name="VI-ARN"):
    """
    Configures and returns a logger that outputs to both console and a file.

    Args:
        log_file (str, optional): Path to the log file. If None, no file handler is added.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
