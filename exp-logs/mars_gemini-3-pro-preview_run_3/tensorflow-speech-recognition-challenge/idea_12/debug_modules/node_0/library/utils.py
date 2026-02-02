import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricMonitor:
    """
    Tracks the running average of a metric (loss, accuracy, etc.) during an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            val (float): The value to add.
            n (int): The number of samples associated with this value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns the current average as a string with full precision."""
        return str(self.avg)


def save_checkpoint(model, optimizer, scheduler, epoch, score, path):
    """
    Saves the model, optimizer, and scheduler states to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch number.
        score (float): The validation score (e.g., accuracy).
        path (str): The file path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }

    torch.save(state, path)


def load_checkpoint(model, optimizer=None, scheduler=None, path=None, device="cpu"):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        path (str): The file path of the checkpoint.
        device (str): The device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        tuple: (epoch, score) from the checkpoint. Returns (0, 0.0) if path is invalid.
    """
    if path is None or not os.path.exists(path):
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    # Handle DataParallel wrapping if necessary, though usually handled outside
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)
