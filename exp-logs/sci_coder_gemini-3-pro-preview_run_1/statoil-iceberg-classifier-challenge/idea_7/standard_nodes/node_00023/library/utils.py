import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filepath):
    """
    Saves the model state (or any other state object) to the specified filepath.
    Ensures the directory exists before saving.

    Args:
        state (dict or object): The object to save (usually model.state_dict()).
        filepath (str): The full path where the file should be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(model, filepath, device=None):
    """
    Loads a model state dictionary from the specified filepath into the provided model.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filepath (str): The path to the checkpoint file.
        device (torch.device, optional): The device to map the location to.
                                         Defaults to CUDA if available, else CPU.

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Handle different saving formats (raw state_dict vs dict wrapper)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load into model
    model.load_state_dict(state_dict)

    return model
