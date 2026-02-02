import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    score: float,
    path: str,
):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): The current training epoch.
        score (float): The validation score associated with this checkpoint.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "score": score,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None,
):
    """
    Loads a checkpoint and restores the model, optimizer, and scheduler states.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to restore.
        device (torch.device, optional): The device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary containing metadata (epoch, score).
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
