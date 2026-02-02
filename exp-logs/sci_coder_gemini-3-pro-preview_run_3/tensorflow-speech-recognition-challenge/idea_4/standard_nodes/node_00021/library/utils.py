import os
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def save_checkpoint(state, filename=config.MODEL_SAVE_PATH):
    """
    Saves the model and training state to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(
    model, filename=config.MODEL_SAVE_PATH, optimizer=None, device="cpu"
):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): Device to map the location to (cpu or cuda).

    Returns:
        dict: The full loaded checkpoint dictionary (useful for retrieving epoch, loss, etc.).
              Returns None if the file does not exist.
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
