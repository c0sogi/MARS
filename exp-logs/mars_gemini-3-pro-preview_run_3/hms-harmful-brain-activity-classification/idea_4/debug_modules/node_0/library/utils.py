import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main", log_file: str = None):
    """
    Configures and returns a logger that writes to console and optionally to a file.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file. If None, defaults to a file in WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "training.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def kl_divergence_score(y_true, y_pred, epsilon: float = 1e-15):
    """
    Calculates the Kullback-Leibler Divergence between ground truth and predictions.
    Supports both numpy arrays and torch tensors.

    Metric Formula: KL(P || Q) = sum(P * log(P / Q))

    Args:
        y_true: Ground truth probabilities (P). Shape (N, Classes).
        y_pred: Predicted probabilities (Q). Shape (N, Classes).
        epsilon: Small value to prevent log(0).

    Returns:
        float: The average KL divergence score.
    """
    # Convert numpy to torch if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure inputs are float
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Clip predictions for numerical stability
    y_pred = torch.clamp(y_pred, min=epsilon, max=1.0 - epsilon)

    # PyTorch KLDivLoss expects input to be Log-Probabilities (log(Q))
    # and target to be Probabilities (P).
    # reduction='batchmean' mathematically aligns with the KL definition averaged over samples.
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    # Calculate loss
    # Input: Log(Predicted Probabilities)
    # Target: True Probabilities
    loss = loss_fn(torch.log(y_pred), y_true)

    return loss.item()
