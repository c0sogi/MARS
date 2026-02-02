import os
import sys
import random
import logging
import numpy as np
import torch
import pydicom
from library.config import Config


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "train", log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger that outputs to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file. If None, defaults to 'train.log' in WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if logger is already configured
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        # Using a simple format. Full precision is handled by the printing logic in the main loop,
        # but the logger passes the message through.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def load_dicom(path: str):
    """
    Safely loads a DICOM file using pydicom.

    Args:
        path (str): The file path to the DICOM image.

    Returns:
        pydicom.dataset.FileDataset: The loaded DICOM dataset object, or None if loading fails.
    """
    try:
        # stop_before_pixels=False ensures we read the whole file including pixel data
        # force=True allows reading files missing the DICOM preamble
        dicom = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        return dicom
    except Exception as e:
        print(f"Error loading DICOM file {path}: {e}")
        return None
