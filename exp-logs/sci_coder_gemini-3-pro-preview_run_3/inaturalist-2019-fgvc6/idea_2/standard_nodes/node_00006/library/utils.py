import os
import sys
import random
import logging
import numpy as np
import torch
from library import config


def seed_everything(seed: int = config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Determines the available hardware device for computation.

    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_logger(name: str = "main_logger") -> logging.Logger:
    """
    Configures and returns a logger instance that writes to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # Only add handler if not already added to avoid duplicate logs
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        # Simple format: Time - Level - Message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
