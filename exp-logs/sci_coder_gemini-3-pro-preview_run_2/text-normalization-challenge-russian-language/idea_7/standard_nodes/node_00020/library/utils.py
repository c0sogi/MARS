import os
import sys
import random
import logging
import re
import unicodedata
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "logger", log_file: str = None, level: int = logging.INFO):
    """
    Sets up a logger with the specified name, file, and level.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, logs only to console.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def is_semiotic(text: str) -> bool:
    """
    Checks if the text contains digits or Latin characters, indicating it might
    need special normalization (semiotic token).

    Args:
        text (str): The input text token.

    Returns:
        bool: True if text contains digits or Latin characters, False otherwise.
    """
    if not text:
        return False

    # Check for digits
    if re.search(Config.REGEX_DIGIT, text):
        return True

    # Check for Latin characters
    if re.search(Config.REGEX_LATIN, text):
        return True

    return False


def normalize_text(text: str) -> str:
    """
    Performs basic text normalization such as Unicode normalization and whitespace stripping.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    if not isinstance(text, str):
        return str(text)

    # Normalize unicode characters to NFC form
    text = unicodedata.normalize("NFC", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
