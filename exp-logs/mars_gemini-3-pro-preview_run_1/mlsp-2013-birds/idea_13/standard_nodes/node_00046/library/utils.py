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
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="model_best.pth"
):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filename, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The path to the checkpoint file or filename in WORKING_DIR.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str, optional): Device to map location to. Defaults to Config.DEVICE.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if device is None:
        device = Config.DEVICE

    # Resolve path: check if absolute/relative exists, otherwise check in WORKING_DIR
    if os.path.exists(filename):
        filepath = filename
    else:
        filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state
    # Handle case where checkpoint is just state_dict or a full dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
