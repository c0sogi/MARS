import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "LPADF_Logger") -> logging.Logger:
    """
    Configures and returns a logger instance for standard logging.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Check if handlers already exist to prevent duplicate logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger
