import os
import shutil
import torch
from library.config import seed_everything


def set_seed(seed):
    """
    Sets the random seed for reproducibility by wrapping the library config function.

    Args:
        seed (int): The seed value to set for random, numpy, and torch.
    """
    seed_everything(seed)


def save_checkpoint(state, is_best, checkpoint_dir, fold_idx):
    """
    Saves the training checkpoint to the specified directory.

    If is_best is True, copies the checkpoint to a 'model_best' file.

    Args:
        state (dict): Dictionary containing model state_dict, optimizer state, epoch, etc.
        is_best (bool): Boolean indicating if this is the best model so far.
        checkpoint_dir (str): Directory path to save the checkpoint.
        fold_idx (int): The current fold index (used for naming).
    """
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define filenames
    filename = f"checkpoint_fold_{fold_idx}.pth"
    filepath = os.path.join(checkpoint_dir, filename)

    # Save the state dictionary
    torch.save(state, filepath)

    # If this is the best model, create a copy
    if is_best:
        best_filename = f"model_best_fold_{fold_idx}.pth"
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint into the provided model and optional optimizer.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        start_epoch (int): The epoch to resume from (default 0).
        best_score (float): The best validation score recorded (default inf).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Extract metadata
    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score
