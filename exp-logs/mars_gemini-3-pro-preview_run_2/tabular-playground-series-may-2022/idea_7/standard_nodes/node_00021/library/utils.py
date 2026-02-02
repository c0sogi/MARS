import os
import torch
from library.config import seed_everything


def save_checkpoint(model, optimizer, scheduler, epoch, metric, filepath):
    """
    Saves the model checkpoint including state dicts for model, optimizer, and scheduler.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (can be None).
        epoch (int): Current epoch.
        metric (float): Validation metric (e.g., AUC).
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metric": metric,
    }

    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary containing epoch and metric info.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if (
        optimizer is not None
        and "optimizer" in checkpoint
        and checkpoint["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    if (
        scheduler is not None
        and "scheduler" in checkpoint
        and checkpoint["scheduler"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def log_metric(name, value):
    """
    Logs a metric with full precision as required.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
