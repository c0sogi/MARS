import os
import sys
import gc
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_resources():
    """
    Aggressively cleans up GPU memory and system resources to prevent OOM.
    """
    # 1. Clear exception tracebacks which hold stack frames and local variables
    if hasattr(sys, "last_traceback"):
        del sys.last_traceback

    # 2. Force full garbage collection
    gc.collect()

    # 3. Release GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_logger(name, log_file=None):
    """
    Creates and configures a logger.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def pf1_score(labels, preds, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        labels (np.array or torch.Tensor): Ground truth binary labels (0 or 1).
        preds (np.array or torch.Tensor): Predicted probabilities (0 to 1).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert tensors to numpy if necessary
    if hasattr(labels, "cpu"):
        labels = labels.detach().cpu().numpy()
    if hasattr(preds, "cpu"):
        preds = preds.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    labels = np.asarray(labels).astype(float)
    preds = np.asarray(preds).astype(float)

    # Flatten arrays
    labels = labels.flatten()
    preds = preds.flatten()

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(preds * labels)
    p_tp = np.sum(preds * labels)

    # Calculate Probabilistic False Positives (pFP)
    # pFP = Sum(preds * (1 - labels))
    p_fp = np.sum(preds * (1 - labels))

    # Calculate Ground Truth Positives (TP + FN)
    # This is simply the sum of the binary labels
    gt_positives = np.sum(labels)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(preds * labels + preds - preds * labels) = Sum(preds)
    denominator_prec = p_tp + p_fp
    p_precision = p_tp / (denominator_prec + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (gt_positives + epsilon)

    # Calculate Probabilistic F1
    # pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return float(p_f1)
