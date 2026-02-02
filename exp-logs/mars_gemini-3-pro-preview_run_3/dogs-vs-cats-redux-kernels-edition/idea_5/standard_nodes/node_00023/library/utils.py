import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filename: str, is_best: bool = False):
    """
    Saves the model state to a file. If is_best is True, copies the file to 'model_best.pth'.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
        is_best (bool): Whether this checkpoint represents the best model so far.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)

    # If it's the best model, create a copy
    if is_best:
        dirname = os.path.dirname(filename)
        best_filename = os.path.join(dirname, "model_best.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(
    filename: str, model: torch.nn.Module, optimizer=None, scheduler=None, device=None
):
    """
    Loads a checkpoint into the model, and optionally the optimizer and scheduler.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (str, optional): The device to map the location to (e.g., 'cuda', 'cpu').
                                Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if device is None:
        device = Config.DEVICE

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Handle state_dict loading
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Fix potential DataParallel module. prefix issues
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    # Load optimizer state if provided and present
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided and present
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
