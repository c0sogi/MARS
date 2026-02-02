import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    """
    Worker initialization function for DataLoader to ensure reproducibility.
    Cite Lesson 00011.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """
    Returns the appropriate torch device (cuda or cpu).

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified filename.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device=None
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (torch.device, optional): Device to map the location to. Defaults to current device.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
    """
    if device is None:
        device = get_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    # Handle case where checkpoint is just the state_dict or a full dict containing 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if (
        scheduler is not None
        and isinstance(checkpoint, dict)
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
