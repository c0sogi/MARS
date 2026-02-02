import os
import random
import numpy as np
import torch
from library.config import WORKING_DIR


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The name of the file to save (e.g., 'model_seed_0.pth').
    """
    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)
    # print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
) -> dict:
    """
    Loads a model checkpoint from the working directory.

    Args:
        filename (str): The name of the file to load.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    filepath = os.path.join(WORKING_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
