import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_micro_f1(logits, targets, threshold=0.5):
    """
    Calculates the Micro-averaged F1 score.

    Args:
        logits (torch.Tensor): Raw model outputs (before sigmoid) of shape (N, NumClasses).
        targets (torch.Tensor): Ground truth multi-hot labels of shape (N, NumClasses).
        threshold (float): Probability threshold for binarizing predictions.

    Returns:
        float: The Micro F1 score.
    """
    # Ensure inputs are on CPU and detached
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # Binarize predictions based on threshold
    preds = (probs > threshold).float()

    # Convert to numpy for sklearn
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    # Calculate Micro F1
    # Zero_division=0 handles cases where no labels are predicted/present
    score = f1_score(targets_np, preds_np, average="micro", zero_division=0)

    return score


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model state, optimizer state, scheduler state, and current metrics.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch number.
        score (float): Validation score (e.g., F1) at this checkpoint.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "score": score,
    }

    torch.save(state, filename)


def load_checkpoint(model, path, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model and optionally optimizer/scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (torch.device, optional): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if device is None:
        device = Config.device

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return epoch, score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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
