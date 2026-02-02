import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures CuDNN to be deterministic.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filename: str):
    """
    Saves the model and optimizer state to a file.

    Args:
        state (dict): The state dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = Config.DEVICE,
):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
        device (str): The device to map the checkpoint to (e.g., 'cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found at {filename}")

    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    # We assume the checkpoint has a 'state_dict' key for the model
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict itself
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
