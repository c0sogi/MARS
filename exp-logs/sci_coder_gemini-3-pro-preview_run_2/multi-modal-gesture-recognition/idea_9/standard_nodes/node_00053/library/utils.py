import os
import sys
import logging
import torch
import numpy as np
import random
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config.set_seed method to ensure consistency.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def get_device():
    """
    Checks for CUDA availability and returns the appropriate torch device.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def setup_logger(name="MD-CRCN", log_file=None):
    """
    Configures a logger with a standard format for console and optional file output.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to a log file. If provided, logs will also be written here.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if handler already exists
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Optional)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
