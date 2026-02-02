import os
import torch
import numpy as np
import random
from library.config import Config, seed_everything


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the seed_everything function from the config library.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def get_device():
    """
    Determines and returns the computation device based on availability.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    return torch.device(Config.DEVICE)


def log_message(message):
    """
    Logs a message to the console.

    Args:
        message (str): The message to log.
    """
    print(message)


def save_checkpoint(model, optimizer, epoch, metric, path):
    """
    Saves the model state, optimizer state, and current metrics to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current training epoch.
        metric (float): The validation metric (e.g., loss or AUC) associated with this checkpoint.
        path (str): The file path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "metric": metric,
    }

    torch.save(state, path)
    log_message(f"Checkpoint saved to {path} (Epoch: {epoch}, Metric: {metric})")


def load_checkpoint(model, path, optimizer=None):
    """
    Loads a model checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): The file path of the checkpoint.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into. Defaults to None.

    Returns:
        dict: The full checkpoint dictionary if loaded successfully, else None.
    """
    if not os.path.exists(path):
        log_message(f"No checkpoint found at {path}")
        return None

    device = get_device()
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    log_message(
        f"Checkpoint loaded from {path} (Epoch: {checkpoint.get('epoch')}, Metric: {checkpoint.get('metric')})"
    )

    return checkpoint
