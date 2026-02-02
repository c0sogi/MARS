import os
import random
import shutil
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir, fold_idx):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        fold_idx (int): The current fold index.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define filenames
    filename = f"checkpoint_fold_{fold_idx}.pth"
    filepath = os.path.join(checkpoint_dir, filename)

    # Save the current state
    torch.save(state, filepath)

    # If it is the best model, create a copy
    if is_best:
        best_filename = f"model_best_fold_{fold_idx}.pth"
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally the optimizer and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    # Check if 'state_dict' key exists, otherwise assume the whole checkpoint is the state dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    # Load optimizer state if provided and available
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided and available
    if (
        scheduler is not None
        and isinstance(checkpoint, dict)
        and "scheduler" in checkpoint
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
