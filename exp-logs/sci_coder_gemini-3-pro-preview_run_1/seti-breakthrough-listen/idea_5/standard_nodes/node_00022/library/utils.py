import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_score(y_true, y_pred):
    """
    Calculates the ROC AUC score.

    Args:
        y_true (np.array): Ground truth labels (binary).
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: The Area Under the ROC Curve.
    """
    # Handle edge case where only one class is present in the batch/set
    if len(np.unique(y_true)) < 2:
        # Returning 0.5 as a neutral score, or could raise an error depending on preference.
        # For training loops, usually we just want to avoid crashing.
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model checkpoint to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): Current epoch number.
        score (float): Validation score (ROC AUC).
        filename (str): Path to save the checkpoint.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "score": score,
    }
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        tuple: (epoch, score) from the checkpoint.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)
