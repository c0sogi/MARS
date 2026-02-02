import os
import random
import numpy as np
import torch
import logging
import sys
import nltk
from library.config import SEED, WORKING_DIR


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_file_name="train.log"):
    """
    Sets up a logger to write to both console and a file.

    Args:
        log_file_name (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_path = os.path.join(WORKING_DIR, log_file_name)

    # Create logger
    logger = logging.getLogger("GestureRecognition")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def calculate_levenshtein(predictions, targets):
    """
    Calculates the Levenshtein error rate metric.

    Metric = (Sum of Levenshtein distances) / (Total number of ground truth gestures)

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        targets (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure sequences are lists
        p = list(pred_seq) if pred_seq is not None else []
        t = list(target_seq) if target_seq is not None else []

        # Calculate Levenshtein distance for this sequence pair
        # nltk.edit_distance works for lists of integers as well as strings
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length
