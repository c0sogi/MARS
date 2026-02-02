import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to disk. If the current model is the best one found so far,
    it creates a copy with a specific 'model_best' filename.

    Args:
        state (dict): A dictionary containing the model state, optimizer state, epoch, etc.
        is_best (bool): Flag indicating if this is the best model based on validation metrics.
        fold (int): The current cross-validation fold index.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Define the standard checkpoint filename
    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    # Save the state dictionary
    torch.save(state, filepath)

    # If this is the best model, save a copy
    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads a model checkpoint from a file.

    Args:
        filepath (str): The path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary containing state_dict, optimizer, etc.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    # Load the checkpoint to the appropriate device defined in Config
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and available in the checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
