import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN backend.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including optimizer state and training metadata.

    Args:
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        loss (float): Validation loss or primary metric.
        path (str): File path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from the specified path.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The PyTorch model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        dict: The full checkpoint dictionary containing epoch, loss, etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
