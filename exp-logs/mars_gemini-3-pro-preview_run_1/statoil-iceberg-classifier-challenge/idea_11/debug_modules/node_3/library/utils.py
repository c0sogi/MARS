import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed=config.RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms ensure reproducibility but may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Saves the model training checkpoint to the specified file.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, epoch, etc.
        filename (str): Full path to save the checkpoint file.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device=config.DEVICE):
    """
    Loads a model checkpoint from the specified file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to (e.g., 'cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load checkpoint
    # Note: weights_only=False is used to allow loading the dictionary structure containing non-weight data
    # If using a very old torch version that doesn't support weights_only, this kwarg might need removal,
    # but given the context of modern ML packages, explicit False is safer for full checkpoints.
    try:
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    # Check if the checkpoint is a dict with a 'state_dict' key or just the weights
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available in checkpoint
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
