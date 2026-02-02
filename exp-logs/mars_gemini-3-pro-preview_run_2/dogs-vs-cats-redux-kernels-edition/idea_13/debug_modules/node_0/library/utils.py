import os
import torch
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Delegates to the Config class to maintain a single source of truth for seeding logic.
    """
    Config.set_seed(seed)


def save_checkpoint(state, filename):
    """
    Saves the model state (and optionally optimizer/scheduler state) to a file.
    Automatically creates the parent directory if it does not exist.

    Args:
        state (dict): The state dictionary to save.
        filename (str): The path where the checkpoint will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model, and optionally into the optimizer and scheduler.
    Handles both full checkpoint dictionaries (with 'model_state_dict' keys) and
    simple state dictionaries (weights only).

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (object, optional): Scheduler to load state into.
        device (str, optional): Device to map the location to. Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if device is None:
        device = Config.DEVICE

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Handle case where checkpoint is a full training state (dict with keys)
    # vs just the model weights (dict of tensors)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume it's a raw state dict (e.g. from model souping)
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and present in checkpoint
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided and present in checkpoint
    if (
        scheduler is not None
        and isinstance(checkpoint, dict)
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def calc_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric for binary classification.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Use sklearn's log_loss. We explicitly provide labels=[0, 1] to ensure
    # correct calculation even if a batch contains only one class.
    return log_loss(y_true, y_pred, labels=[0, 1])
