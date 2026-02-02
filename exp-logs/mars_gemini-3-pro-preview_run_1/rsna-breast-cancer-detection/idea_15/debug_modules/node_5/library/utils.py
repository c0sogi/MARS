import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pf1_score(labels, preds, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task.

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        labels (np.ndarray or torch.Tensor): Ground truth binary labels (0 or 1).
        preds (np.ndarray or torch.Tensor): Predicted probabilities (0.0 to 1.0).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The calculated pF1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    # Flatten arrays to ensure 1D vectors
    labels = labels.flatten()
    preds = preds.flatten()

    # Probabilistic True Positives: sum(prob * label)
    # Only contributes when label is 1
    p_tp = np.sum(preds * labels)

    # Probabilistic False Positives: sum(prob * (1 - label))
    # Only contributes when label is 0
    p_fp = np.sum(preds * (1 - labels))

    # Total Positives (TP + FN): sum(labels)
    # This is the count of actual positive cases
    total_positives = np.sum(labels)

    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equal to sum(preds)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # pF1 calculation
    numerator = 2 * p_precision * p_recall
    denominator = p_precision + p_recall

    # Handle case where denominator is 0
    if denominator == 0:
        return 0.0

    pf1 = numerator / (denominator + epsilon)

    return pf1


def get_logger(name="cancer_detection"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # Check if handlers already exist to avoid duplicate logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid double printing if root is configured
        logger.propagate = False

    return logger
