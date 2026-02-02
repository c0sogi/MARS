import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
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
    torch.cuda.manual_seed_all(seed)  # Safe for multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="CR-WBN", log_file=None):
    """
    Configures and returns a logger instance that writes to stdout and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to 'run.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        # Ensure the working directory exists before defining the log path
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "run.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = (
        False  # Prevent propagation to root logger to avoid double printing
    )

    # Check if handlers already exist to avoid adding duplicates on multiple calls
    if not logger.handlers:
        # Define the format for the logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        try:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(
                f"Warning: Failed to create log file handler at {log_file}. Error: {e}"
            )

        # Stream Handler (Stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
