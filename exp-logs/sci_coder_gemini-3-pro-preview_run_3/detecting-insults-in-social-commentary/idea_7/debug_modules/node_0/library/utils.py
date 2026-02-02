import os
import sys
import random
import numpy as np
import torch
import codecs
import logging


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text found in the dataset.
    Example: "\\xe2\\x80\\x9c" -> "“"

    Args:
        text: The input text (string or NaN).

    Returns:
        str: The decoded string.
    """
    if text is None:
        return ""

    # Handle NaN values often present in pandas Series
    if isinstance(text, float) and np.isnan(text):
        return ""

    try:
        # Ensure input is a string
        s = str(text)
        # Decode using unicode_escape to interpret backslash sequences
        decoded = codecs.decode(s, "unicode_escape")
        return decoded
    except Exception:
        # In case of malformed strings, return the original representation
        return str(text)


def get_logger(name: str = "training"):
    """
    Creates and configures a logger that outputs to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
