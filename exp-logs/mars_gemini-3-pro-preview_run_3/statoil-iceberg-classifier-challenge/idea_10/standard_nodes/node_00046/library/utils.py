import os
import shutil
import torch
import numpy as np
from library.config import Config, set_seed


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


def save_checkpoint(state, is_best, fold):
    """
    Saves the model checkpoint to the configured checkpoint directory.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best validation performance so far.
        fold (int): The current fold index (used for naming the file).
    """
    # Ensure directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"model_best_fold_{fold}.pth"
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, optimizer=None, filename=None):
    """
    Loads a model checkpoint from a file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): The path to the checkpoint file.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary, or None if not found.
    """
    if not filename or not os.path.isfile(filename):
        return None

    # Load checkpoint to the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Load model state dict
    # Handle cases where state_dict might be nested under 'state_dict' or 'model_state_dict'
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state dict if provided
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
