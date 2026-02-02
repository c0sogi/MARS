import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets fixed random seeds for Python's random module, NumPy, and PyTorch to ensure
    reproducibility of results.

    Args:
        seed (int): The seed value to use. Defaults to the value in Config.SEED.
    """
    # Python random module
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Ensure deterministic behavior for cuDNN backend
        # This is critical for reproducing results with CNNs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file_path: str = None):
    """
    Configures the root logger to output messages to the console and optionally to a file.

    Args:
        log_file_path (str, optional): The path to the log file. If None, logging is only to console.

    Returns:
        logging.Logger: The configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file_path:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
