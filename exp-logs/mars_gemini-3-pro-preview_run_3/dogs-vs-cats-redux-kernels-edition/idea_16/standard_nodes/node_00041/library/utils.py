import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the torch device based on configuration and availability.

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device(Config.DEVICE)


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified filename.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (torch.device, optional): Device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary (excluding state_dicts loaded into objects),
              or None if the file is not found.
    """
    if not os.path.isfile(filename):
        # We generally prefer silent execution, but a missing checkpoint is critical info
        # usually handled by the caller, but returning None signals failure.
        return None

    if device is None:
        device = get_device()

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # Handle different saving conventions (wrapped in 'state_dict', 'model_state_dict', or raw)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the log loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1.

    Returns:
        float: The log loss value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to avoid log(0) error (numerical stability)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = log_loss(y_true, y_pred)
    return loss
