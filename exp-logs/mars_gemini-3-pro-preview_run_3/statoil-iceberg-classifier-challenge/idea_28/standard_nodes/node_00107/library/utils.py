import os
import random
import shutil
import numpy as np
import torch
from library import config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, fold_idx):
    """
    Saves the training checkpoint to the directory specified in config.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        fold_idx (int): The current fold index (used for naming).
    """
    # Ensure checkpoint directory exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Define filenames
    filename = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_fold_{fold_idx}.pth")
    best_filename = os.path.join(
        config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
    )

    # Save the current state
    torch.save(state, filename)

    # If this is the best model, create a copy
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(model, filename, optimizer=None):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filename (str): The path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary containing metadata (epoch, best_score, etc.).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found at: {filename}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(filename, map_location=config.DEVICE)

    # Load model weights
    # We use strict=False optionally if needed, but strict=True is better for reproducibility
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
