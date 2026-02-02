import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import ProjectConfig


def seed_everything(seed: int = ProjectConfig.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to ProjectConfig.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str = "project"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger. Defaults to "project".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it's configured and return it
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def generate_config_hash():
    """
    Retrieves the unique configuration hash from ProjectConfig.
    This hash is based on critical hyperparameters and feature sets
    and is used for cache invalidation.

    Returns:
        str: MD5 hash string of the current configuration.
    """
    return ProjectConfig.get_config_hash()
