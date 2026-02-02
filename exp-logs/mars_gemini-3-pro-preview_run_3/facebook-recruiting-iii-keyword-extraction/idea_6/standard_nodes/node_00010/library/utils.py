import os
import random
import sys
import re
import logging
import numpy as np
import torch
from sklearn.metrics import f1_score
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can be slower, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text):
    """
    Preprocesses raw text strings by lowercasing and removing non-alphanumeric characters.
    Replaces non-alphanumeric characters with spaces to preserve word boundaries.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Convert to lowercase
    text = text.lower()

    # Replace any character that is not alphanumeric or whitespace with a space
    # This handles punctuation, special symbols, etc.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple whitespace characters into a single space and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_f1_score(y_true, y_pred, average="micro"):
    """
    Calculates the F1 score using Scikit-Learn.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average (str): The averaging strategy for F1 score. Defaults to 'micro'.

    Returns:
        float: The calculated F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def get_logger(name="pipeline"):
    """
    Sets up and returns a logger instance that writes to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        # Use a simple message format
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
