import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility by calling the centralized
    seed_everything function from the config module.
    """
    seed_everything(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: ROC AUC score.
    """
    # Handle case where only one class is present in the batch/set to avoid errors
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for mixup
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename=None):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): Current epoch number.
        score (float): Validation score (ROC AUC).
        filename (str, optional): Path to save the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
    """
    if filename is None:
        filename = Config.MODEL_SAVE_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(
    model, optimizer=None, scheduler=None, filename=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        filename (str, optional): Path to the checkpoint file. Defaults to Config.MODEL_SAVE_PATH.
        device (str): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (containing epoch, score, etc.), or None if file not found.
    """
    if filename is None:
        filename = Config.MODEL_SAVE_PATH

    if not os.path.exists(filename):
        return None

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
