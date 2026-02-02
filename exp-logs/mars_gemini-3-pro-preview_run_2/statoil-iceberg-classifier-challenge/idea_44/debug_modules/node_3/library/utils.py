import os
import shutil
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model state to a file.
    If is_best is True, copies the file to a 'best' version in the same directory.

    Args:
        state (dict): The state dictionary to save (model weights, optimizer, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): The full path to save the checkpoint.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the checkpoint
    torch.save(state, filepath)

    # If this is the best model, create a copy with a specific suffix
    if is_best:
        base, ext = os.path.splitext(filepath)
        best_filepath = f"{base}_best{ext}"
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads model weights and optionally optimizer state from a checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, loss, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Load checkpoint to the correct device
    checkpoint = torch.load(filepath, map_location=torch.device(config.DEVICE))

    # Determine if checkpoint is a full state dict wrapper or just weights
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Standard checkpoint format with metadata
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        # Assume direct state dict
        model.load_state_dict(checkpoint)

    return checkpoint
