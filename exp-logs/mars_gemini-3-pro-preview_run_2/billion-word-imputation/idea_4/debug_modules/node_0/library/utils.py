import os
import sys
import random
import logging
import numpy as np
import torch
import pandas as pd
import csv
import nltk
from library.config import Config


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in CuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "project_logger", log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger that outputs to console and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
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
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def clean_text(text: str) -> str:
    """
    Sanitizes input text.
    Performs whitespace normalization.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)

    # Normalize whitespace: collapse multiple spaces into one and strip leading/trailing
    return " ".join(text.split())


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein distance between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: The Levenshtein edit distance.
    """
    return nltk.edit_distance(s1, s2)


def save_submission(ids: list, sentences: list, output_path: str) -> None:
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,"sentence"
    1,"Predicted sentence..."

    Args:
        ids (list): List of sentence IDs.
        sentences (list): List of predicted sentences.
        output_path (str): Path to save the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "sentence": sentences})

    # Use QUOTE_NONNUMERIC to quote strings (sentences) but not integers (ids)
    # Double quotes within strings are escaped as "" by default in pandas/csv
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Submission saved to {output_path}")
