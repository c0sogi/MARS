import os
import sys
import random
import logging
import numpy as np
import torch
import nltk
from typing import List, Union


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(
    name: str, log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger with console and optional file handlers.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, only console logging is enabled.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicates if function is called multiple times
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
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_levenshtein(references: List[str], hypotheses: List[str]) -> float:
    """
    Computes the average Levenshtein distance between a list of reference sentences
    and a list of hypothesis sentences.

    Args:
        references (List[str]): List of ground truth sentences.
        hypotheses (List[str]): List of predicted sentences.

    Returns:
        float: The average Levenshtein distance.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: references ({len(references)}) vs hypotheses ({len(hypotheses)})"
        )

    if not references:
        return 0.0

    total_distance = 0
    for ref, hyp in zip(references, hypotheses):
        # nltk.edit_distance computes the Levenshtein distance
        dist = nltk.edit_distance(ref, hyp)
        total_distance += dist

    return total_distance / len(references)


def escape_sentence_for_csv(sentence: str) -> str:
    """
    Formats a sentence for the submission CSV file according to the task requirements:
    - Use double quotes to escape the sentence text.
    - Use two double quotes ("") for double quotes within a sentence.

    Args:
        sentence (str): The raw sentence text.

    Returns:
        str: The escaped and quoted sentence string ready for CSV writing.
    """
    # Escape internal double quotes by replacing " with ""
    escaped_sentence = sentence.replace('"', '""')

    # Enclose the entire sentence in double quotes
    return f'"{escaped_sentence}"'
