import os
import torch
import numpy as np
import random
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility.
    Delegates to the Config class to ensure consistency across the library.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current training epoch.
        metrics (dict or float): Validation metrics or score associated with this checkpoint.
        filepath (str): The destination path for the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create the state dictionary
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
    }

    # Save using torch
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    """
    Loads model and optimizer state from a checkpoint file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device): The device to map the checkpoint data to.

    Returns:
        dict: The loaded checkpoint dictionary if successful, None otherwise.
    """
    if not os.path.exists(filepath):
        return None

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and available
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def print_metrics(epoch, metrics):
    """
    Prints validation metrics with full precision.

    Args:
        epoch (int): The current epoch.
        metrics (dict): A dictionary mapping metric names to values.
    """
    # Using default string formatting to ensure full precision is printed
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    print(f"Epoch {epoch} Metrics: {metrics_str}")


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
