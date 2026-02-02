import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename=Config.MODEL_PATH):
    """
    Saves the model and optimizer state to a file.
    Automatically creates the directory if it does not exist.

    Args:
        state (dict): The state dictionary to save (e.g., {'state_dict': ..., 'optimizer': ...}).
        filename (str): The path where the checkpoint will be saved. Defaults to Config.MODEL_PATH.
    """
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(
    model, optimizer=None, filename=Config.MODEL_PATH, device=Config.DEVICE
):
    """
    Loads the model and optimizer state from a file.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
        filename (str): The path to the checkpoint file. Defaults to Config.MODEL_PATH.
        device (str): The device to map the checkpoint to ('cpu' or 'cuda'). Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary if successful, None otherwise.
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    # Checks for common key names used in state dictionaries
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback: try loading the checkpoint directly if it is the state dict itself
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model state dict: {e}")

    # Load optimizer state if provided
    if optimizer:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
