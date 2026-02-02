import os
import torch
import numpy as np
import random
from library.config import Config, seed_everything


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility using the imported seed_everything function.
    """
    seed_everything(seed)


class MetricTracker:
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # Returns the full precision string representation of the average
        return str(self.avg)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): The current epoch.
        loss (float): The validation loss (or metric) at this checkpoint.
        path (str): Path to save the checkpoint file.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    torch.save(state, path)
    # We do not print here to keep output clean, as per instructions.


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from the specified path.

    Args:
        path (str): Path to the checkpoint file.
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device: The device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (containing epoch, loss, etc.).
        None: If the path does not exist.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
