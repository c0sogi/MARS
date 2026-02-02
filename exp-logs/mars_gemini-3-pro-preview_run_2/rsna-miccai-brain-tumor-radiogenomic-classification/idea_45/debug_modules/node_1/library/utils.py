import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED constant.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="root", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to a file where logs should be saved.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (optional)
        if log_file:
            ensure_dir(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_dir(path):
    """
    Ensures that the directory exists. If it doesn't, it creates it.

    Args:
        path (str): The directory path to check/create.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def join_path(*args):
    """
    Joins path components intelligently.

    Args:
        *args: Path components.

    Returns:
        str: The joined path.
    """
    return os.path.join(*args)
