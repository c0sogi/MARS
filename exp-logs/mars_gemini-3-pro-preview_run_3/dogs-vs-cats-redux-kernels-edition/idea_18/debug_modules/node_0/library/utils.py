import os
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import WORKING_DIR, SUBMISSION_PATH


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def get_logger(name: str, log_file: str = None):
    """
    Configures and returns a logger instance that logs to console and optionally a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_checkpoint_path(model_name: str, fold: int) -> str:
    """
    Generates the filepath for saving or loading a model checkpoint.

    Args:
        model_name (str): The unique name of the model configuration.
        fold (int): The fold number.

    Returns:
        str: Full path to the checkpoint file.
    """
    filename = f"{model_name}_fold_{fold}.pth"
    return os.path.join(WORKING_DIR, filename)


def save_submission(ids, probabilities, output_path=SUBMISSION_PATH):
    """
    Saves the prediction results to a CSV file in the required format.

    Args:
        ids (list or np.array): List of image IDs.
        probabilities (list or np.array): List of predicted probabilities.
        output_path (str): Path where the submission CSV will be saved.
    """
    df = pd.DataFrame({"id": ids, "label": probabilities})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
