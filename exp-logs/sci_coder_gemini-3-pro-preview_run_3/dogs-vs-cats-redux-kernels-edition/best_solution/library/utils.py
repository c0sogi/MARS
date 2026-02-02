import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library import config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model and training state to a file in the working directory.

    Args:
        state (dict): Dictionary containing model state_dict, optimizer state, etc.
        filename (str): Name of the file to save.
    """
    filepath = os.path.join(config.WORKING_DIR, filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(filename, model=None, optimizer=None, device="cpu"):
    """
    Loads a checkpoint from the working directory.

    Args:
        filename (str): Name of the checkpoint file.
        model (torch.nn.Module, optional): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    # Construct path; check if it's relative to WORKING_DIR or absolute
    filepath = os.path.join(config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        if os.path.exists(filename):
            filepath = filename
        else:
            print(f"Checkpoint file not found: {filepath}")
            return None

    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        # Handle different naming conventions for state dicts
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume the checkpoint itself is the state dict
            model.load_state_dict(checkpoint)

    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities for class 1.

    Returns:
        float: The calculated log loss.
    """
    # Clip predictions to avoid log(0) error, though sklearn usually handles this.
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return log_loss(y_true, y_pred)
