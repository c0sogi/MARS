import sys
import os
import logging
from library.config import set_seed


def setup_logger(log_file_path, name="logger", level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.

    Args:
        log_file_path (str): Path to the log file.
        name (str): Name of the logger.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear handlers if they exist to avoid duplication in interactive environments
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Prevent propagation to root logger to avoid double logging
    logger.propagate = False

    return logger
