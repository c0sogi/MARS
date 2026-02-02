import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config


def calculate_f1_score(y_true, y_pred, threshold=0.5):
    """
    Calculates the Micro-averaged F1 score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).
        threshold (float): Threshold for binarizing predictions.

    Returns:
        float: Micro F1 score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    y_pred_bin = (y_pred > threshold).astype(int)

    # Calculate Micro-F1 score
    return f1_score(y_true, y_pred_bin, average="micro")


def optimize_threshold(y_true, y_pred, num_steps=100):
    """
    Finds the optimal decision threshold for Micro F1 score via linear search.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
        num_steps (int): Number of steps in the linear search (between 0 and 1).

    Returns:
        tuple: (best_threshold, best_score)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    best_threshold = 0.5
    best_score = -1.0

    # Search range from 0.01 to 0.99 to avoid edge cases
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        score = calculate_f1_score(y_true, y_pred, threshold=thresh)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves a model checkpoint to the working directory.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        score (float): Validation score (F1).
        filename (str): Name of the file to save.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "score": score,
    }

    torch.save(state, filepath)


def load_checkpoint(model, filename, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Name of the checkpoint file (relative to WORKING_DIR or absolute).
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str): Device to map location to.

    Returns:
        tuple: (epoch, score)
    """
    # Determine full path
    if os.path.exists(filename):
        filepath = filename
    else:
        filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided and available
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return epoch, score
