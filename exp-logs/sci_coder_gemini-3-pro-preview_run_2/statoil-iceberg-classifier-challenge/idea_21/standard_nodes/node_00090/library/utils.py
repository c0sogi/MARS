import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to 'run.log' in Config.IDEA_DIR.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.IDEA_DIR, "run.log")

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
