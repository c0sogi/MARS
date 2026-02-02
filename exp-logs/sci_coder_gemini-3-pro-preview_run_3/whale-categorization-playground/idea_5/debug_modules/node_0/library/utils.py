import os
import shutil
import torch
import numpy as np
import random
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates the actual implementation to Config.setup_reproducibility to avoid duplication.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.setup_reproducibility(seed)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to record (e.g., batch loss).
            n (int): The weight of the value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filepath=None):
    """
    Saves the model checkpoint to the specified filepath.
    If is_best is True, also copies the file to 'model_best.pth' in the same directory.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Flag indicating if this is the best model so far.
        filepath (str, optional): Path to save the checkpoint.
                                  Defaults to 'checkpoint.pth' in Config.WORKING_DIR.
    """
    if filepath is None:
        filepath = os.path.join(Config.WORKING_DIR, "checkpoint.pth")

    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(directory, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Loads model weights (and optionally optimizer/scheduler state) from a checkpoint file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (object, optional): The learning rate scheduler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary if successful, else None.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at '{filepath}'")
        return None

    print(f"Loading checkpoint from '{filepath}'")
    # Load to the configured device (CPU or GPU)
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Handle different state dict keys for robustness
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided and available
    if scheduler is not None:
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
