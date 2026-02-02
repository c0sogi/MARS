import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=42):
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


def get_logger(name="training"):
    """
    Creates a logger that outputs to both console (stdout) and a log file.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

        # File Handler
        log_path = os.path.join(Config.WORKING_DIR, f"{name}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def probabilistic_f1(y_true, y_pred):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true: Ground truth labels (binary 0 or 1). Can be a numpy array or torch Tensor.
        y_pred: Predicted probabilities [0, 1]. Can be a numpy array or torch Tensor.

    Returns:
        float: The calculated pF1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D structure
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    epsilon = 1e-7

    # pTP: Sum of probabilities for actual positive cases
    # pTP = sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # pFP: Sum of probabilities for actual negative cases
    # pFP = sum((1 - y_true) * y_pred)
    p_fp = np.sum((1 - y_true) * y_pred)

    # TP + FN: Total number of actual positives
    total_positives = np.sum(y_true)

    # pPrecision = pTP / (pTP + pFP)
    # Note: The denominator (pTP + pFP) is equivalent to sum(y_pred)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # pF1 Calculation
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): If True, saves an additional copy as 'best_model.pth'.
        filename (str): The filename for the checkpoint.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Save the current checkpoint
    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        torch.save(state, best_path)
