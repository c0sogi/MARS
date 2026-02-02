import os
import random
import numpy as np
import torch
from library.config import SEED, DEVICE


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Saves the model training state to a file.

    Args:
        state (dict): Dictionary containing model state, optimizer state, epoch, etc.
        filename (str): Path to save the checkpoint file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=DEVICE):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to (cpu or cuda).

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # Handle cases where the model might be wrapped in DataParallel (keys start with 'module.')
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Assume direct state dict save if key missing

    # Check for 'module.' prefix mismatch
    if list(state_dict.keys())[0].startswith("module.") and not hasattr(
        model, "module"
    ):
        # Remove 'module.' prefix
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
