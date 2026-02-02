import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, filepath):
    """
    Saves the model and training state to a checkpoint file.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device, optional): Device to map the location to. Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if device is None:
        device = Config.DEVICE

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_array(data, filepath):
    """
    Saves a numpy array to a file, ensuring the directory exists.
    Used for caching processed data.

    Args:
        data (np.ndarray): Data to save.
        filepath (str): Path to save the .npy file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(filepath, data)


def load_array(filepath):
    """
    Loads a numpy array from a file.

    Args:
        filepath (str): Path to the .npy file.

    Returns:
        np.ndarray: Loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Numpy file not found: {filepath}")
    return np.load(filepath)
