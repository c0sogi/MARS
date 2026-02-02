import os
import random
import sys
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The random seed value. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "root", level: int = logging.INFO):
    """
    Sets up a logger that outputs to sys.stdout.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Use a simple format to keep output clean as requested
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
