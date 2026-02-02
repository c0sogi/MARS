import os
import sys
import logging
from library.config import Config, seed_everything


def get_logger(name=__name__, log_filename="train.log"):
    """
    Configures and returns a logger that writes to both stdout and a log file.

    Args:
        name (str): The name of the logger.
        log_filename (str): The name of the log file to be saved in Config.WORK_DIR.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Define the format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        # Ensure the working directory exists
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        log_path = os.path.join(Config.WORK_DIR, log_filename)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
