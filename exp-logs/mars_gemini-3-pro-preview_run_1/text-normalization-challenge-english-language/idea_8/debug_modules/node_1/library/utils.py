import os
import torch
from library.config import Config, set_seed


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves the model, optimizer, and scheduler states to a checkpoint file.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): The current epoch.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        path (str): Path to the checkpoint file.
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device: The device to map the checkpoint to.

    Returns:
        int: The epoch saved in the checkpoint, or 0 if not found/loaded.
    """
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)


def compute_metrics(predictions, references):
    """
    Computes the exact match accuracy between predictions and references.

    Args:
        predictions (list[str]): List of predicted normalized text strings.
        references (list[str]): List of ground truth normalized text strings.

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) must have the same length."
        )

    if len(references) == 0:
        return 0.0

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(references)
