import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name=__name__, log_file=None):
    """
    Configures and returns a logger that prints to stdout and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def probabilistic_f1(y_true, y_pred, beta=1):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = (1 + beta^2) * (pPrecision * pRecall) / ((beta^2 * pPrecision) + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).
        beta (float): Beta value for F-score (default 1.0).

    Returns:
        float: The probabilistic F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # pTP = Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # pFP = Sum((1 - y_true) * y_pred)
    p_fp = np.sum((1 - y_true) * y_pred)

    # Total Positives (TP + FN) is simply the number of positive ground truth cases
    total_positives = np.sum(y_true)

    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP simplifies to Sum(y_pred)
    sum_pred = np.sum(y_pred)

    if sum_pred == 0:
        p_precision = 0.0
    else:
        p_precision = p_tp / sum_pred

    if total_positives == 0:
        p_recall = 0.0
    else:
        p_recall = p_tp / total_positives

    # Calculate F1
    if p_precision + p_recall == 0:
        pf1 = 0.0
    else:
        pf1 = (
            (1 + beta**2)
            * (p_precision * p_recall)
            / ((beta**2 * p_precision) + p_recall)
        )

    return pf1


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, filename):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): Current epoch.
        best_score (float): Best validation score achieved so far.
        filename (str): Path to save the checkpoint.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, filename, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map location to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Load model weights
    # Use strict=False if necessary, but typically we want strict loading for reproducibility
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return epoch, best_score
