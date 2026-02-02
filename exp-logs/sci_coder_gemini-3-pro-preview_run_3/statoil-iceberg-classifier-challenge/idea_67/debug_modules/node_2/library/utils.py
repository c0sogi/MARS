import os
import sys
import random
import logging
import warnings
import numpy as np
import torch
from library.config import SEED, WORKING_DIR


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # CUDA Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_filename="execution.log", log_level=logging.INFO):
    """
    Configures the logging system to write to a file and the console.

    Args:
        log_filename (str): Name of the log file. Saved in WORKING_DIR.
        log_level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("idea_67")
    logger.setLevel(log_level)

    # Clear existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    log_path = os.path.join(WORKING_DIR, log_filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def disable_warnings():
    """
    Suppresses warnings to keep the output clean.
    """
    warnings.filterwarnings("ignore")
    # Suppress TensorFlow/other library C++ level warnings if applicable
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
