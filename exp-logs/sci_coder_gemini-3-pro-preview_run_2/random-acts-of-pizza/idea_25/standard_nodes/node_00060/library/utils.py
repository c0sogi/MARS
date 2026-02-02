import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch settings
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "execution", log_file: str = None, level: int = logging.INFO
):
    """
    Configures and returns a logger instance that writes to stdout and an optional file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, only logs to stdout.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
