import os
import sys
import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import CHECKPOINT_DIR, seed_everything


def get_logger(name, log_file=None):
    """
    Creates and configures a logger.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger already exists
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Stream Handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_scores: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Handle case where y_scores might be logits or probabilities
    # Assuming probabilities for AUC, but if logits are passed,
    # the rank order is preserved so AUC is same (usually).

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # This can happen if there is only one class in y_true
        return 0.5


def save_checkpoint(state, filename="checkpoint.pth", is_best=False):
    """
    Saves the model state to the checkpoint directory.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, etc.
        filename (str): Name of the checkpoint file.
        is_best (bool): Whether this is the best model so far.
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        checkpoint_path (str): Full path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch/score).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle case where checkpoint is just state_dict or full dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
