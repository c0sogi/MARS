import os
import sys
import random
import logging
import numpy as np
import torch
import nltk
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "root", log_file: str = None, level: int = logging.INFO):
    """
    Configures and returns a logger instance that writes to stdout and an optional file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, no file handler is added.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler (Optional)
    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_levenshtein(reference: str, hypothesis: str) -> int:
    """
    Calculates the Levenshtein distance between a reference sentence and a hypothesis sentence.

    Args:
        reference (str): The correct original sentence.
        hypothesis (str): The predicted sentence.

    Returns:
        int: The Levenshtein edit distance.
    """
    return nltk.edit_distance(reference, hypothesis)


def compute_average_levenshtein(references: list, hypotheses: list) -> float:
    """
    Computes the average Levenshtein distance over a batch of sentences.

    Args:
        references (list of str): List of ground truth sentences.
        hypotheses (list of str): List of predicted sentences.

    Returns:
        float: The average Levenshtein distance.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"References list length ({len(references)}) must match hypotheses list length ({len(hypotheses)})."
        )

    if len(references) == 0:
        return 0.0

    total_distance = 0
    for ref, hyp in zip(references, hypotheses):
        total_distance += calculate_levenshtein(ref, hyp)

    return total_distance / len(references)
