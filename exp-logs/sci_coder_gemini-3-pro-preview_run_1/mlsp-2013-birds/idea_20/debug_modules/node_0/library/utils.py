import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for cuDNN backend.

    Args:
        seed (int, optional): The seed value to use. Defaults to Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filepath):
    """
    Saves the checkpoint dictionary to the specified file path.
    Creates the directory if it does not exist.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint from the specified file path into the provided objects.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str, optional): Device to map the location to. Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if device is None:
        device = Config.DEVICE

    checkpoint = torch.load(filepath, map_location=device)

    # Handle different conventions for saving state dicts
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
