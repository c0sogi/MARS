import os
import sys
import random
import logging
import codecs
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text sequences (e.g., \\u00e9 -> é) found in the dataset.
    Handles NaN/None values by returning an empty string.

    Args:
        text: The input text (string or NaN).

    Returns:
        str: The decoded text or empty string.
    """
    if pd.isna(text):
        return ""
    try:
        # Ensure input is string
        s = str(text)
        # Decode unicode_escape characters to get proper unicode representation
        return codecs.decode(s, "unicode_escape")
    except Exception:
        # Return original string if decoding fails
        return str(text)


def get_logger(name: str = "main"):
    """
    Configures and returns a logger instance.
    Logs are output to both standard output and a file in the working directory defined in Config.

    Args:
        name (str): The name of the logger. Defaults to "main".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        # Use Config to determine output directory
        os.makedirs(Config.working_dir, exist_ok=True)
        log_file = os.path.join(Config.working_dir, f"{name}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
