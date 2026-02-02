import os
import sys
import logging
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Delegates the actual implementation to the Config class to ensure consistency.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def get_logger(name="training", log_file=None):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to 'train.log'
                                  inside Config.WORK_DIR.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Determine log file path
    if log_file is None:
        # Ensure work dir exists
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORK_DIR, "train.log")
    else:
        # Ensure directory for the specific log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to avoid duplication
    if not logger.handlers:
        # Define format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create file handler for logger: {e}")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger
