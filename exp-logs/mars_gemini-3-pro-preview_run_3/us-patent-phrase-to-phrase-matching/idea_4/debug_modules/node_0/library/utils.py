import os
import sys
import random
import logging
import numpy as np
import torch
from scipy.stats import pearsonr
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure deterministic behavior.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="main", log_filename="train.log"):
    """
    Creates and configures a logger that writes to both stdout and a file.

    Args:
        name (str): The name of the logger.
        log_filename (str): The name of the log file to save in Config.output_dir.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the output directory exists
    Config.create_output_dir()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the logger is retrieved again
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    log_path = os.path.join(Config.output_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_metrics(predictions, labels):
    """
    Computes the Pearson correlation coefficient between predictions and labels.

    Args:
        predictions (np.ndarray or torch.Tensor): The model predictions.
        labels (np.ndarray or torch.Tensor): The ground truth labels.

    Returns:
        dict: A dictionary containing the 'pearson' correlation score.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Flatten the arrays to ensure they are 1D
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Compute Pearson correlation
    # pearsonr returns (statistic, p-value)
    pearson_score, _ = pearsonr(predictions, labels)

    return {"pearson": pearson_score}
