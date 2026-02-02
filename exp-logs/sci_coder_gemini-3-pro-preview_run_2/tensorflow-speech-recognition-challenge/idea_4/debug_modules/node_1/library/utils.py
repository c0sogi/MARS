import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path=Config.BEST_MODEL_PATH,
    scheduler=None,
    extra_info=None,
):
    """
    Saves the model checkpoint including model state, optimizer state, and metadata.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch number.
        loss (float): The validation loss or primary metric.
        path (str): The file path to save the checkpoint. Defaults to Config.BEST_MODEL_PATH.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler.
        extra_info (dict, optional): Any additional information to save in the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if extra_info is not None:
        state.update(extra_info)

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from the specified path.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): The device to map the checkpoint to (e.g., 'cuda' or 'cpu'). Defaults to Config.DEVICE.

    Returns:
        dict: The full checkpoint dictionary loaded from the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
