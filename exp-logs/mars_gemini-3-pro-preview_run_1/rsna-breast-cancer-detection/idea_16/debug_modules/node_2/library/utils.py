import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Where:
    pTP = sum(y_true * y_pred) -> Sum of probs for actual positives
    pFP = sum((1 - y_true) * y_pred) -> Sum of probs for actual negatives
    TP + FN = sum(y_true) -> Total count of actual positives

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels (0 or 1).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (0 to 1).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Probabilistic True Positives (pTP)
    p_tp = np.sum(y_true * y_pred)

    # Calculate Probabilistic False Positives (pFP)
    p_fp = np.sum((1 - y_true) * y_pred)

    # Calculate Total Positives (TP + FN)
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # Denominator is sum of all predicted probabilities: p_tp + p_fp = sum(y_pred)
    precision_denom = p_tp + p_fp
    p_precision = p_tp / (precision_denom + epsilon)

    # Calculate pRecall
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    f1_denom = p_precision + p_recall
    p_f1 = 2 * (p_precision * p_recall) / (f1_denom + epsilon)

    return p_f1


def get_logger(name, log_file=None):
    """
    Creates and configures a logger.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, logs to working directory.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file is None:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "training.log")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
