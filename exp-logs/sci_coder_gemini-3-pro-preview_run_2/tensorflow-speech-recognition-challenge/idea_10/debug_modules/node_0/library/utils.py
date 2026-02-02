import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def save_checkpoint(state, is_best, filepath=Config.LAST_MODEL_PATH):
    """
    Saves the current model state to a checkpoint file.
    If is_best is True, also copies the file to Config.BEST_MODEL_PATH.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.).
        is_best (bool): Whether the current checkpoint represents the best model so far.
        filepath (str): The path to save the checkpoint. Defaults to Config.LAST_MODEL_PATH.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, Config.BEST_MODEL_PATH)


def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    filepath=Config.BEST_MODEL_PATH,
    device=Config.DEVICE,
):
    """
    Loads model weights and optional optimizer/scheduler state from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The learning rate scheduler to load state into.
        filepath (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at: {filepath}")
        return None

    print(f"Loading checkpoint from: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    # Handle both 'state_dict' key and direct state dict saving
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Returns the device specified in Config.
    """
    return torch.device(Config.DEVICE)
