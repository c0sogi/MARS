import os
import sys
import random
import logging
import numpy as np


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is not installed or not required for this run
        pass


def setup_logger(name: str = "pipeline", level: int = logging.INFO):
    """
    Configures and returns a logger instance with a standard format.
    Ensures handlers are not duplicated if the logger is fetched multiple times.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add formatter to handler
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    # Ensure the level is set correctly if the logger existed but level changed
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    return logger
