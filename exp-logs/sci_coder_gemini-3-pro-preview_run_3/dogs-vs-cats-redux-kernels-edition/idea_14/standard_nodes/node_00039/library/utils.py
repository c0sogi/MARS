import os
import random
import shutil
import numpy as np
import torch
from library.config import SEED, WORKING_DIR, DEVICE


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
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


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name=None, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        # Returns full precision string representation
        return f"{self.val}"


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="model_best.pth"
):
    """
    Saves the training state to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
    """
    # Ensure working directory exists (redundant if config does it, but safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    filepath = os.path.join(WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filename, optimizer=None, scheduler=None):
    """
    Loads a checkpoint from the working directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The filename of the checkpoint to load.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary.
    """
    filepath = os.path.join(WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    checkpoint = torch.load(filepath, map_location=DEVICE)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    elif scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
