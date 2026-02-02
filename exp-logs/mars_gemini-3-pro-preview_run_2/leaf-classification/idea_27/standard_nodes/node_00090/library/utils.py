import os
import sys
import random
import numpy as np
import logging
import warnings
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, Numpy, and Torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Attempt to set seed for PyTorch if installed
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logger(name="leaf_classification", log_file=None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to a file to log to.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        # Simple formatting: just the message, as complex timestamps aren't requested
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def suppress_warnings():
    """
    Suppresses standard warnings and library-specific warnings to keep output clean.
    """
    warnings.filterwarnings("ignore")
    # Suppress TensorFlow logs if present
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
