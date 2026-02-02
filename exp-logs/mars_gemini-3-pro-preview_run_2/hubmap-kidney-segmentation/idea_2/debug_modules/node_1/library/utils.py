import os
import sys
import random
import numpy as np
import torch
import logging
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The format is pairs of values: start position and run length.
    Pixels are numbered from top to bottom, then left to right (Fortran order).
    1-based indexing is used.

    Args:
        img (np.ndarray): Binary mask of shape (Height, Width).
                          1 indicates masked pixel, 0 indicates background.

    Returns:
        str: RLE string 'start length start length ...'
    """
    # Flatten in column-major order (Fortran-style) as per requirements
    pixels = img.flatten(order="F")

    # Pad with zeros at start and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def setup_logger(log_file=None):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        log_file (str, optional): Path to the log file.
                                  If None, defaults to 'train.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Simple format for console to avoid clutter
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
