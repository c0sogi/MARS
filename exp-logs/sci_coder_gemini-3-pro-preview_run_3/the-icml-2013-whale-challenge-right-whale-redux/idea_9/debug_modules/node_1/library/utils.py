import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import TrainConfig


def set_seed(seed=TrainConfig.seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the torch device based on configuration.
    """
    return torch.device(TrainConfig.device)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def calculate_metrics(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities.

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # robust check for single-class batches to avoid sklearn errors
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, filename):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        best_score (float): Best validation score so far.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_score": best_score,
    }
    torch.save(state, filename)


def load_checkpoint(model, filename, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (optional): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch/score).
    """
    if device is None:
        device = get_device()

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer
        and "optimizer_state_dict" in checkpoint
        and checkpoint["optimizer_state_dict"] is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
