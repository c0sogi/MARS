import logging
import os
import sys
from library.config import seed_everything


def get_logger(log_file):
    """
    Configures and returns a logger that writes to both a file and standard output.

    Args:
        log_file (str): The file path where logs should be saved.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File Handler: Logs detailed information including timestamp
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream Handler: Logs messages to stdout (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    # Use a simpler format for console output to keep it clean
    stream_formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger
