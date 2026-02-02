import os
import sys
import random
import logging
import numpy as np
import torch


def seed_everything(seed: int):
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str):
    """
    Creates and configures a logger that writes to both a file and the console (stdout).

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the classification accuracy.

    Args:
        output (torch.Tensor): Logits or probabilities from the model (Batch, Num_Classes).
        target (torch.Tensor): Ground truth labels (Batch).

    Returns:
        float: The accuracy ratio (correct / total).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        _, preds = torch.max(output, dim=1)
        correct = (preds == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy
